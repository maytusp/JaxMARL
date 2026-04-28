import argparse
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxmarl.environments.overcooked_v2.common import Actions, Direction, DIR_TO_VEC
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts as layouts
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
# python -m jaxmarl.environments.overcooked_v2.interactive --layout cramped_room --agent_view_size 2 --debug


class OvercookedToMTransform:
    """
    Build the partner's fictitious egocentric observation from the ego agent's
    observation. The partner's heading defines the crop and the final alignment,
    while the available content is limited to the ego agent's current visual field.
    """

    def __init__(self, agent_view_size=2, agent_features_len=9):
        self.agent_view_size = agent_view_size
        self.agent_features_len = agent_features_len

    def _rotate_spatial(self, obs, direction):
        def rotate_0(x):
            return x

        def rotate_90(x):
            return jnp.rot90(x, k=1, axes=(0, 1))

        def rotate_180(x):
            return jnp.rot90(x, k=2, axes=(0, 1))

        def rotate_270(x):
            return jnp.rot90(x, k=3, axes=(0, 1))

        rotation_idx = jnp.select(
            [
                direction == Direction.UP,
                direction == Direction.DOWN,
                direction == Direction.LEFT,
            ],
            [0, 2, 3],
            default=1,
        )

        return lax.switch(
            rotation_idx,
            (rotate_0, rotate_90, rotate_180, rotate_270),
            obs,
        )

    def _rotate_direction_vec(self, vec, direction):
        def rotate_0(v):
            return v

        def rotate_90(v):
            return jnp.array([v[1], -v[0]], dtype=v.dtype)

        def rotate_180(v):
            return -v

        def rotate_270(v):
            return jnp.array([-v[1], v[0]], dtype=v.dtype)

        rotation_idx = jnp.select(
            [
                direction == Direction.UP,
                direction == Direction.DOWN,
                direction == Direction.LEFT,
            ],
            [0, 2, 3],
            default=1,
        )

        return lax.switch(
            rotation_idx,
            (rotate_0, rotate_90, rotate_180, rotate_270),
            vec,
        )

    def _extract_pose(self, pos_channel, dir_channels):
        view_span = 2 * self.agent_view_size + 1
        flat_idx = jnp.argmax(pos_channel.reshape(-1))
        row = flat_idx // view_span
        col = flat_idx % view_span
        direction = jnp.argmax(dir_channels[row, col])
        return row, col, direction

    def _absolute_to_local_direction(self, abs_direction, ego_abs_direction):
        local_vec = self._rotate_direction_vec(
            DIR_TO_VEC[abs_direction], ego_abs_direction
        )

        return jnp.select(
            [
                jnp.all(local_vec == DIR_TO_VEC[Direction.UP]),
                jnp.all(local_vec == DIR_TO_VEC[Direction.DOWN]),
                jnp.all(local_vec == DIR_TO_VEC[Direction.LEFT]),
            ],
            [Direction.UP, Direction.DOWN, Direction.LEFT],
            default=Direction.RIGHT,
        )

    def _crop_front_window(self, grid, row, col, local_direction):
        view_size = self.agent_view_size
        window_shape = (2 * view_size + 1, 2 * view_size + 1, grid.shape[-1])

        def crop_up(x):
            padded = jnp.pad(
                x,
                ((2 * view_size, 0), (view_size, view_size), (0, 0)),
                constant_values=0,
            )
            return jax.lax.dynamic_slice(padded, (row, col, 0), window_shape)

        def crop_down(x):
            padded = jnp.pad(
                x,
                ((0, 2 * view_size), (view_size, view_size), (0, 0)),
                constant_values=0,
            )
            return jax.lax.dynamic_slice(padded, (row, col, 0), window_shape)

        def crop_right(x):
            padded = jnp.pad(
                x,
                ((view_size, view_size), (0, 2 * view_size), (0, 0)),
                constant_values=0,
            )
            return jax.lax.dynamic_slice(padded, (row, col, 0), window_shape)

        def crop_left(x):
            padded = jnp.pad(
                x,
                ((view_size, view_size), (2 * view_size, 0), (0, 0)),
                constant_values=0,
            )
            return jax.lax.dynamic_slice(padded, (row, col, 0), window_shape)

        crop_idx = jnp.select(
            [
                local_direction == Direction.UP,
                local_direction == Direction.DOWN,
                local_direction == Direction.LEFT,
            ],
            [0, 1, 3],
            default=2,
        )

        return lax.switch(
            crop_idx,
            (crop_up, crop_down, crop_right, crop_left),
            grid,
        )

    def __call__(self, self_obs):
        original_shape = self_obs.shape
        flat_obs = self_obs.reshape(-1, *original_shape[-3:])
        transformed_flat = jax.vmap(self._transform_single_frame)(flat_obs)
        return transformed_flat.reshape(original_shape)

    def _transform_single_frame(self, grid):
        self_pos_channel = grid[..., 0]
        self_dir_channels = grid[..., 1:5]
        partner_pos_channel = grid[..., self.agent_features_len]
        partner_dir_channels = grid[
            ..., self.agent_features_len + 1 : self.agent_features_len + 5
        ]

        _, _, ego_abs_dir = self._extract_pose(self_pos_channel, self_dir_channels)
        partner_row, partner_col, partner_abs_dir = self._extract_pose(
            partner_pos_channel, partner_dir_channels
        )
        partner_present = jnp.max(partner_pos_channel)

        partner_local_dir = self._absolute_to_local_direction(
            partner_abs_dir, ego_abs_dir
        )
        crop = self._crop_front_window(
            grid, partner_row, partner_col, partner_local_dir
        )
        crop = self._rotate_spatial(crop, partner_local_dir)

        self_features = crop[..., 0 : self.agent_features_len]
        partner_features = crop[
            ..., self.agent_features_len : 2 * self.agent_features_len
        ]
        rest_features = crop[..., 2 * self.agent_features_len :]
        transformed = jnp.concatenate(
            [partner_features, self_features, rest_features], axis=-1
        )

        return transformed * partner_present



class InteractiveOvercookedV2:

    def __init__(self, layout, agent_view_size=None, no_jit=False, debug=False):
        self.debug = debug
        self.no_jit = no_jit

        self.env = OvercookedV2(layout=layout, agent_view_size=agent_view_size)
        self.tom_transform = OvercookedToMTransform(
            agent_view_size=self.env.agent_view_size
            if self.env.agent_view_size is not None
            else 2
        )
        self.viz = OvercookedV2Visualizer()

    def run(self, key):
        self.key = key
        with jax.disable_jit(self.no_jit):
            self._run()

    def _run(self):
        self._reset()

        self.viz.window.reg_key_handler(self._handle_input)
        self.viz.show(block=True)

    def _handle_input(self, event):
        if self.debug:
            print("Pressed", event.key)

        ACTION_MAPPING = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.up,
            "down": Actions.down,
            " ": Actions.interact,
            "tab": Actions.stay,
        }

        match event.key:
            case "escape":
                self.viz.window.close()
                return
            case "backspace":
                self._reset()
                return
            case key if key in ACTION_MAPPING:
                action = ACTION_MAPPING[key]
            case key:
                print(f"Key {key} not recognized")
                return

        self._step(action)

    def _redraw(self):
        self.viz.render(self.state, agent_view_size=self.env.agent_view_size)

    def _print_obs_debug(self, obs):
        np.set_printoptions(linewidth=200, threshold=np.inf, suppress=True)
        for agent_id in range(self.env.num_agents-1):
            agent_name = f"agent_{agent_id}"
            agent_obs = obs[agent_name]
            transformed_obs = self.tom_transform(agent_obs)

            print(f"{agent_name} observation shape: {agent_obs.shape}")
            print(
                f"{agent_name} observation [C,H,W]:\n",
                np.asarray(jnp.transpose(agent_obs, (2, 0, 1))[18:20]),
            )
            print(f"{agent_name} transformed observation shape: {transformed_obs.shape}")
            print(
                f"{agent_name} transformed observation [C,H,W]:\n",
                np.asarray(jnp.transpose(transformed_obs, (2, 0, 1))[18:20]),
            )

    def _reset(self):
        self.key, key = jax.random.split(self.key)
        obs, state = jax.jit(self.env.reset)(key)
        self.state = state

        if self.debug:
            print(f"t={state.time}: reset observation")
            self._print_obs_debug(obs)

        self._redraw()

    def _step(self, action):
        self.key, subkey = jax.random.split(self.key)

        actions = {f"agent_{i}": jnp.array(action) for i in range(self.env.num_agents)}
        if self.debug:
            print("Actions: ", actions)

        obs, state, reward, done, info = jax.jit(self.env.step_env)(
            subkey, self.state, actions
        )
        self.state = state
        print(f"t={state.time}: reward={reward['agent_0']}, done = {done['__all__']}")

        if self.debug:
            self._print_obs_debug(obs)
            print("Reward: ", reward)
            print("Shaped reward: ", info["shaped_reward"])

        if done["__all__"]:
            self._reset()
        else:
            self._redraw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout", type=str, help="Overcooked layout", default="cramped_room"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=0,
    )
    parser.add_argument(
        "--agent_view_size",
        type=int,
        help="Number of cells in the agent view. If not provided, the agent will see the whole grid.",
    )
    parser.add_argument(
        "--no_jit",
        default=False,
        help="Disable JIT compilation",
        action="store_true",
    )
    parser.add_argument(
        "--debug", default=False, help="Debug mode", action="store_true"
    )
    args = parser.parse_args()

    if len(args.layout) == 0:
        raise ValueError("You must provide a layout.")
    layout = layouts[args.layout]

    interactive = InteractiveOvercookedV2(
        layout=layout,
        agent_view_size=args.agent_view_size,
        no_jit=args.no_jit,
        debug=args.debug,
    )

    key = jax.random.PRNGKey(args.seed)
    interactive.run(key)
