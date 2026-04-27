import argparse
import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.overcooked_v2.common import Actions
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts as layouts
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

class OvercookedToMTransform:
    """
    Transforms an sekf observation into a partner observation (other-stream)
    via Translation and Channel Swapping.
    """
    def __init__(self, agent_view_size=2, agent_features_len=9):
        self.agent_view_size = agent_view_size
        self.agent_features_len = agent_features_len

    def __call__(self, self_obs):
        # self_obs can be [B, H, W, C] or [Time, Batch, H, W, C]
        original_shape = self_obs.shape
        
        # Flatten all leading batch dimensions so we have [N, H, W, C]
        flat_obs = self_obs.reshape(-1, *original_shape[-3:])
        
        # Apply the transform to the flat batch
        transformed_flat = jax.vmap(self._transform_single_frame)(flat_obs)
    
        # Reshape back to the original batch/time dimensions
        return transformed_flat.reshape(original_shape)

    def _transform_single_frame(self, grid):
        """
        1. Find Partner -> 2. Pad Grid -> 3. Dynamic Slice (Translate) -> 4. Swap Channels
        """
        V = 2 * self.agent_view_size + 1 # e.g., 5 for a view_size of 2
        
        # --- 1. FIND THE PARTNER ---
        # The partner's position is the first channel of the "Other Agent" block.
        # Ego block: grid[..., 0 : agent_features_len]
        # Partner block: grid[..., agent_features_len : 2 * agent_features_len]
        partner_pos_channel = grid[..., self.agent_features_len]
        
        # Find local (r, c) of the partner within the self's current view
        flat_idx = jnp.argmax(partner_pos_channel.flatten())
        p_r = flat_idx // V
        p_c = flat_idx % V
        
        # Check if the partner is actually visible (to handle edge cases where they are out of view)
        partner_present = jnp.max(partner_pos_channel)
        
        # --- 2. PAD THE GRID ---
        # Pad by `agent_view_size` on all spatial sides with 0.
        # This naturally handles the blindspots, filling unknown areas with 0 natively.
        padded_grid = jnp.pad(
            grid, 
            ((self.agent_view_size, self.agent_view_size), 
             (self.agent_view_size, self.agent_view_size), 
             (0, 0)), 
            constant_values=0
        )
        
        # --- 3. DYNAMIC CROP (TRANSLATE) ---
        # To center the partner at (agent_view_size, agent_view_size),
        # the start indices of the crop on the padded grid are simply (p_r, p_c).
        crop = jax.lax.dynamic_slice(
            padded_grid,
            (p_r, p_c, 0),
            (V, V, grid.shape[-1])
        )
        
        # --- 4. SWAP CHANNELS ---
        # In the other-stream, the Partner becomes the "Ego" and the Ego becomes the "Partner".
        self_features = crop[..., 0 : self.agent_features_len]
        partner_features = crop[..., self.agent_features_len : 2 * self.agent_features_len]
        rest_features = crop[..., 2 * self.agent_features_len :]
        
        otherstream_view = jnp.concatenate([partner_features, self_features, rest_features], axis=-1)
        
        # Mask out the result if the partner wasn't in the self agent's view to begin with
        return otherstream_view * partner_present


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
        for agent_id in range(self.env.num_agents):
            agent_name = f"agent_{agent_id}"
            agent_obs = obs[agent_name]
            transformed_obs = self.tom_transform(agent_obs)

            print(f"{agent_name} observation shape: {agent_obs.shape}")
            print(
                f"{agent_name} observation [C,H,W]:\n",
                np.asarray(jnp.transpose(agent_obs, (2, 0, 1))),
            )
            print(f"{agent_name} transformed observation shape: {transformed_obs.shape}")
            print(
                f"{agent_name} transformed observation [C,H,W]:\n",
                np.asarray(jnp.transpose(transformed_obs, (2, 0, 1))),
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
