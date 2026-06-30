import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from jaxmarl.environments.overcooked_v2.common import Direction, DIR_TO_VEC

class OvercookedTransform(nn.Module):
    """
    Transforms a self observation into a partner observation (other-stream)
    via Translation and Channel Swapping.
    """
    agent_view_size: int = 2
    
    # This depends on your layout's number of ingredients.
    # For 2 ingredients: 1 (pos) + 4 (dir) + 6 (inv encoding) = 11 channels
    agent_features_len: int = 9

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
    


class OvercookedHeadAlignedTransform:
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
