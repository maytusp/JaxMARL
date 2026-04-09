import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict, Optional, Tuple
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, OvercookedV2LogWrapper
from jaxmarl.environments import overcooked_v2_layouts
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import functools

import jax
import jax.numpy as jnp
from flax import linen as nn

from .rim import DenseModularCell

import math

class OvercookedToMTransform(nn.Module):
    """
    Transforms an sekf observation into a partner observation (other-stream)
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



class TokenEncoder(nn.Module):
    token_dim: int
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, obs):
        # obs: [..., H, W, C]
        x = nn.Conv(
            features=64,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        x = self.activation(x)

        x = nn.Conv(
            features=self.token_dim,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        # flatten spatial map into tokens
        *leading, h, w, c = x.shape
        tokens = x.reshape(*leading, h * w, c)  # [..., N, D]
        return tokens


class ASTScannedGRU(nn.Module):
    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        # carry: [B, H], x = (inputs, resets)
        inputs, resets = x
        batch_size = inputs.shape[0]
        init_carry = self.initialize_carry(batch_size, self.hidden_size)
        carry = jnp.where(resets[:, None], init_carry, carry)
        carry, y = nn.GRUCell(features=self.hidden_size)(carry, inputs)
        return carry, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class MultiHeadTokenAttention(nn.Module):
    token_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, query_vec, tokens, gate_logits=None):
        """
        query_vec: [T, B, D]
        tokens:    [T, B, N, D]
        gate_logits: [T, B, N] or None
        """
        head_dim = self.token_dim // self.num_heads
        assert self.token_dim % self.num_heads == 0, "token_dim must divide num_heads"

        q = nn.Dense(self.token_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="q_proj")(query_vec)
        k = nn.Dense(self.token_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="k_proj")(tokens)
        v = nn.Dense(self.token_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="v_proj")(tokens)

        T, B, N, _ = k.shape

        q = q.reshape(T, B, self.num_heads, head_dim)                 # [T,B,H,Dh]
        k = k.reshape(T, B, N, self.num_heads, head_dim)             # [T,B,N,H,Dh]
        v = v.reshape(T, B, N, self.num_heads, head_dim)

        # [T,B,H,N]
        logits = jnp.einsum("tbhd,tbnhd->tbhn", q, k) / math.sqrt(head_dim)

        if gate_logits is not None:
            logits = logits + gate_logits[:, :, None, :]  # broadcast over heads

        attn = nn.softmax(logits, axis=-1)                # [T,B,H,N]
        attended = jnp.einsum("tbhn,tbnhd->tbhd", attn, v) # [T,B,H,Dh]
        attended = attended.reshape(T, B, self.token_dim)  # [T,B,D]

        mean_attn = attn.mean(axis=2)  # [T,B,N]
        return attended, mean_attn, logits

# V1
# class ActorCriticASTRNN(nn.Module):
#     action_dim: Sequence[int]
#     config: Dict

#     @nn.compact
#     def __call__(self, hidden, x):
#         """
#         hidden = (schema_state, prev_summary)
#           schema_state: [B, Hs]
#           prev_summary: [B, Dtok]
#         x = (obs, dones, prev_actions_onehot)
#           obs: [T,B,H,W,C]
#           dones: [T,B]
#           prev_actions_onehot: [T,B,A]
#         """
#         obs, dones, prev_actions_onehot = x

#         activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
#         token_dim = self.config["AST_TOKEN_DIM"]
#         schema_dim = self.config["AST_SCHEMA_DIM"]
#         num_heads = self.config["AST_NUM_HEADS"]

#         schema_state, prev_summary = hidden

#         obs_self = obs
#         if self.config.get("AST_USE_OTHER_STREAM", True):
#             obs_other = OvercookedToMTransform()(obs)
#         else:
#             obs_other = None

#         token_encoder = TokenEncoder(token_dim=token_dim, activation=activation, name="token_encoder")

#         tokens_self = token_encoder(obs_self)  # [T,B,N,D]

#         if obs_other is not None:
#             tokens_other = token_encoder(obs_other)
#         else:
#             tokens_other = tokens_self

#         self_bias = self.param("self_stream_bias", nn.initializers.zeros, (1, 1, 1, token_dim))
#         other_bias = self.param("other_stream_bias", nn.initializers.zeros, (1, 1, 1, token_dim))
#         tokens_self = tokens_self + self_bias
#         tokens_other = tokens_other + other_bias
#         tokens = jnp.concatenate([tokens_self, tokens_other], axis=-2)  # [T,B,2N,D]


#         # summary input for schema
#         pooled_tokens = tokens.mean(axis=-2)  # [T,B,D]

#         schema_inputs = [pooled_tokens, prev_summary[None, :, :].repeat(obs.shape[0], axis=0)]
#         if self.config.get("AST_USE_PREV_ACTION", True):
#             schema_inputs.append(prev_actions_onehot)

#         schema_input = jnp.concatenate(schema_inputs, axis=-1)  # [T,B,*]

#         schema_rnn = ASTScannedGRU(hidden_size=schema_dim, name="schema_rnn")
#         schema_state, schema_out = schema_rnn(schema_state, (schema_input, dones))  # [T,B,Hs]

#         # schema query for attention
#         query_vec = nn.Dense(
#             token_dim,
#             kernel_init=orthogonal(1.0),
#             bias_init=constant(0.0),
#             name="schema_to_query",
#         )(schema_out)

#         # soft gate over tokens
#         gate_logits = nn.Dense(
#             features=tokens.shape[-2],
#             kernel_init=orthogonal(0.01),
#             bias_init=constant(0.0),
#             name="schema_gate",
#         )(schema_out)  # [T,B,N_tokens]

#         attended_feat, attn_weights, raw_logits = MultiHeadTokenAttention(
#             token_dim=token_dim,
#             num_heads=num_heads,
#             name="ast_attention",
#         )(query_vec, tokens, gate_logits=gate_logits)

#         attended_feat = nn.LayerNorm(name="attended_ln")(attended_feat)

#         pred_attended_feat = nn.Dense(
#             token_dim,
#             kernel_init=orthogonal(1.0),
#             bias_init=constant(0.0),
#             name="schema_predictor",
#         )(schema_out)

#         pred_attended_feat = activation(pred_attended_feat)
#         pred_attended_feat = nn.Dense(
#             token_dim,
#             kernel_init=orthogonal(1.0),
#             bias_init=constant(0.0),
#             name="schema_predictor_out",
#         )(pred_attended_feat)

#         if self.config.get("AST_USE_SCHEMA_IN_POLICY", True):
#             policy_feat = jnp.concatenate([attended_feat, schema_out], axis=-1)
#         else:
#             policy_feat = attended_feat

#         actor_h = nn.Dense(
#             self.config["FC_DIM_SIZE"],
#             kernel_init=orthogonal(2),
#             bias_init=constant(0.0),
#             name="actor_fc1",
#         )(policy_feat)
#         actor_h = activation(actor_h)
#         actor_logits = nn.Dense(
#             self.action_dim,
#             kernel_init=orthogonal(0.01),
#             bias_init=constant(0.0),
#             name="actor_out",
#         )(actor_h)
#         pi = distrax.Categorical(logits=actor_logits)

#         critic_h = nn.Dense(
#             self.config["FC_DIM_SIZE"],
#             kernel_init=orthogonal(2),
#             bias_init=constant(0.0),
#             name="critic_fc1",
#         )(policy_feat)
#         critic_h = activation(critic_h)
#         critic = nn.Dense(
#             1,
#             kernel_init=orthogonal(1.0),
#             bias_init=constant(0.0),
#             name="critic_out",
#         )(critic_h)

#         aux = {
#             "attended_feat": attended_feat,
#             "pred_attended_feat": pred_attended_feat,
#             "gate_logits": gate_logits,
#             "attn_weights": attn_weights,
#         }

#         new_hidden = (
#             schema_state,                 # final recurrent carry after scan
#             attended_feat[-1],            # next prev_summary for rollout step t+1
#         )

#         return new_hidden, pi, jnp.squeeze(critic, axis=-1), aux


# V2
class ActorCriticASTRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        """
        hidden = (schema_state, prev_summary)
          schema_state: [B, Hs]
          prev_summary: [B, Dtok]
        x = (obs, dones, prev_actions_onehot)
          obs: [T,B,H,W,C]
          dones: [T,B]
          prev_actions_onehot: [T,B,A]
        """
        obs, dones, prev_actions_onehot = x

        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        token_dim = self.config["AST_TOKEN_DIM"]
        schema_dim = self.config["AST_SCHEMA_DIM"]
        num_heads = self.config["AST_NUM_HEADS"]

        schema_state, prev_summary = hidden

        obs_self = obs
        if self.config.get("AST_USE_OTHER_STREAM", True):
            obs_other = OvercookedToMTransform()(obs)
        else:
            obs_other = None

        # ---------------------------------------------------------------------
        # MODIFICATION 1: Use standard CNN instead of TokenEncoder
        # ---------------------------------------------------------------------
        cnn_encoder = CNN(
            output_size=token_dim,
            activation=activation,
            name="cnn_encoder"
        )

        h, w, c = obs_self.shape[-3:]
        
        # Flatten time and batch dimensions for Ego CNN pass
        flat_obs_self = obs_self.reshape(-1, h, w, c)
        cnn_out_self = cnn_encoder(flat_obs_self)
        
        # Reshape back to [T, B, token_dim] and expand to [T, B, 1, token_dim]
        # Treat the single vector as a token sequence of length 1 for attention
        cnn_out_self = cnn_out_self.reshape(*obs_self.shape[:-3], -1) 
        tokens_self = jnp.expand_dims(cnn_out_self, axis=-2) 

        if obs_other is not None:
            flat_obs_other = obs_other.reshape(-1, h, w, c)
            cnn_out_other = cnn_encoder(flat_obs_other)
            cnn_out_other = cnn_out_other.reshape(*obs_other.shape[:-3], -1)
            tokens_other = jnp.expand_dims(cnn_out_other, axis=-2)
        else:
            tokens_other = tokens_self

        self_bias = self.param("self_stream_bias", nn.initializers.zeros, (1, 1, 1, token_dim))
        other_bias = self.param("other_stream_bias", nn.initializers.zeros, (1, 1, 1, token_dim))
        tokens_self = tokens_self + self_bias
        tokens_other = tokens_other + other_bias
        
        # Concatenate tokens. With 2 streams, this results in shape [T, B, 2, D]
        tokens = jnp.concatenate([tokens_self, tokens_other], axis=-2)  

        # ---------------------------------------------------------------------
        # MODIFICATION 2: Map to RNN using a linear model instead of tokens.mean()
        # ---------------------------------------------------------------------
        # Flatten the tokens into a single vector dimension [T, B, Num_Tokens * D]
        flat_tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)
        
        # Apply a linear model to project flattened tokens for the Schema RNN
        mapped_tokens = nn.Dense(
            features=token_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="tokens_to_schema_map"
        )(flat_tokens)

        # Assemble schema inputs
        schema_inputs = [mapped_tokens, prev_summary[None, :, :].repeat(obs.shape[0], axis=0)]
        if self.config.get("AST_USE_PREV_ACTION", True):
            schema_inputs.append(prev_actions_onehot)

        schema_input = jnp.concatenate(schema_inputs, axis=-1)  # [T, B, *]

        # Process through Schema RNN
        schema_rnn = ASTScannedGRU(hidden_size=schema_dim, name="schema_rnn")
        schema_state, schema_out = schema_rnn(schema_state, (schema_input, dones))  # [T, B, Hs]

        # Schema query for attention
        query_vec = nn.Dense(
            token_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="schema_to_query",
        )(schema_out)

        # Soft gate over tokens
        gate_logits = nn.Dense(
            features=tokens.shape[-2],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="schema_gate",
        )(schema_out)  # [T, B, N_tokens]

        attended_feat, attn_weights, raw_logits = MultiHeadTokenAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            name="ast_attention",
        )(query_vec, tokens, gate_logits=gate_logits)

        attended_feat = nn.LayerNorm(name="attended_ln")(attended_feat)

        pred_attended_feat = nn.Dense(
            token_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="schema_predictor",
        )(schema_out)

        pred_attended_feat = activation(pred_attended_feat)
        pred_attended_feat = nn.Dense(
            token_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="schema_predictor_out",
        )(pred_attended_feat)

        if self.config.get("AST_USE_SCHEMA_IN_POLICY", True):
            policy_feat = jnp.concatenate([attended_feat, schema_out], axis=-1)
        else:
            policy_feat = attended_feat

        # Actor
        actor_h = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor_fc1",
        )(policy_feat)
        actor_h = activation(actor_h)
        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )(actor_h)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic
        critic_h = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic_fc1",
        )(policy_feat)
        critic_h = activation(critic_h)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(critic_h)

        aux = {
            "attended_feat": attended_feat,
            "pred_attended_feat": pred_attended_feat,
            "gate_logits": gate_logits,
            "attn_weights": attn_weights,
        }

        new_hidden = (
            schema_state,                 # final recurrent carry after scan
            attended_feat[-1],            # next prev_summary for rollout step t+1
        )

        return new_hidden, pi, jnp.squeeze(critic, axis=-1), aux

class ScannedRNN(nn.Module):
    config: Dict = None
    rnn_type: str = None
    
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        
        current_rnn_type = self.rnn_type if self.rnn_type is not None else self.config.get("RNN_TYPE", "gru")
        
        # Dynamically get the batch size of the current step (full batch or minibatch)
        current_batch_size = ins.shape[0]

        if current_rnn_type == "gru":
            hidden_size = ins.shape[-1]
            new_carry = self.initialize_carry(config=self.config, rnn_type="gru", batch_size=current_batch_size)

            rnn_state = jnp.where(resets[:, None], new_carry, rnn_state)
            new_rnn_state, y = nn.GRUCell(features=hidden_size)(rnn_state, ins)
            return new_rnn_state, y

        elif current_rnn_type == "rim":
            new_carry = self.initialize_carry(config=self.config, rnn_type="rim", batch_size=current_batch_size)

            rnn_state = jnp.where(resets[:, None, None], new_carry, rnn_state)

            new_rnn_state, y = DenseModularCell(
                input_size=ins.shape[-1],
                hidden_size=self.config["RIM_HIDDEN_DIM"],
                num_units=self.config["RIM_NUM_UNITS"],
                comm_key_size=self.config.get("RIM_COMM_KEY_SIZE", 128),
                comm_query_size=self.config.get("RIM_COMM_QUERY_SIZE", 128),
                num_comm_heads=self.config.get("RIM_NUM_COMM_HEADS", 4),
            )(rnn_state, ins)

            return new_rnn_state, y

        else:
            raise ValueError(f"Unknown RNN_TYPE: {current_rnn_type}")

    @staticmethod
    def initialize_carry(config, rnn_type=None, batch_size=None):
        if rnn_type is None:
            rnn_type = config.get("RNN_TYPE", "gru")
        
        # Default to NUM_ENVS if no batch_size is provided (e.g. during initial setup)
        if batch_size is None:
            batch_size = config["NUM_ENVS"]
            
        if rnn_type == "rim":
            return DenseModularCell.initialize_carry(
                batch_size=batch_size,
                num_units=config["RIM_NUM_UNITS"],
                hidden_size=config["RIM_HIDDEN_DIM"],
            )
        else:
            hidden_dim = config["GRU_HIDDEN_DIM"]
            cell = nn.GRUCell(features=hidden_dim)
            return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_dim))

class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        if x.ndim == 3:
            x = x[jnp.newaxis, ...] 
            had_no_batch = True
        else:
            had_no_batch = False

        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        
        if had_no_batch:
            x = x.squeeze(0)

        return x


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    rnn_type: str = None

    @nn.compact
    def __call__(self, hidden, x):

        obs, dones = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        h, w, c = obs.shape[-3:]
        # 2. Flatten all leading dimensions (Time and Batch) into one
        # This turns (T, B, H, W, C) -> (T*B, H, W, C)
        flat_obs = obs.reshape(-1, h, w, c)

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh,
        )

        # 3. Apply the CNN to the flattened batch (Guaranteed 4D input)
        embedding = embed_model(flat_obs)

        # 4. Reshape back to the original leading dimensions
        # This turns (T*B, Features) -> (T, B, Features)
        embedding = embedding.reshape(*obs.shape[:-3], -1)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(config=self.config, rnn_type=self.rnn_type)(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    prev_action: jnp.ndarray
    attended_feat: jnp.ndarray
    pred_attended_feat: jnp.ndarray
    gate_logits: jnp.ndarray
    info: jnp.ndarray
    
def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def load_partner_pool(config, checkpoint_names, dummy_params):
    layout_name = config["ENV_KWARGS"]["layout"]
    checkpoints_prefix = config.get("CHECKPOINTS_PREFIX", "./checkpoints/sp_pools")
    pool_dir = os.path.join(checkpoints_prefix, layout_name)

    loaded_params = []
    for ckpt_name in checkpoint_names:
        ckpt_path = os.path.join(pool_dir, ckpt_name)
        with open(ckpt_path, "rb") as f:
            p = flax.serialization.from_bytes(dummy_params, f.read())
            loaded_params.append(p)
            print(f"LOADED: {ckpt_name}")

    return jax.tree_util.tree_map(lambda *args: jnp.stack(args), *loaded_params)

def init_ego_hidden_for_batch(config, batch_size):
    use_tom = config.get("USE_TOM", True)
    param_mode = config.get("PARAM_MODE", "separate")

    if use_tom:
        if param_mode in ["shared", "separate"]:
            return (
                ScannedRNN.initialize_carry(config, batch_size=batch_size),
                ScannedRNN.initialize_carry(config, batch_size=batch_size),
            )
        elif param_mode in ["shared_aggregate", "input_aggregate"]:
            return ScannedRNN.initialize_carry(config, batch_size=batch_size)
        else:
            raise ValueError(f"Unknown PARAM_MODE: {param_mode}")
    else:
        return ScannedRNN.initialize_carry(
            config,
            rnn_type=config.get("RNN_TYPE", "gru"),
            batch_size=batch_size,
        )
def initialize_ast_carry(config, batch_size):
    return (
        ASTScannedGRU.initialize_carry(batch_size, config["AST_SCHEMA_DIM"]),
        jnp.zeros((batch_size, config["AST_TOKEN_DIM"]), dtype=jnp.float32),
    )


def make_zsc_evaluator(config, eval_pop_params):
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = config.get("EVAL_NUM_ENVS", 128)

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    network_ego = ActorCriticASTRNN(
        env.action_space(env.agents[0]).n,
        config=eval_config,
    )
    network_partner = ActorCriticRNN(
        env.action_space(env.agents[1]).n, config=eval_config, rnn_type="gru"
    )

    num_eval_steps = eval_config.get("EVAL_NUM_STEPS", eval_config["NUM_STEPS"])
    num_eval_envs = eval_config["NUM_ENVS"]
    num_eval_episodes = eval_config.get("EVAL_NUM_EPISODES", 8)

    def evaluate_one_partner(rng, ego_params, partner_params):
        def run_one_episode(rng):
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, num_eval_envs)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

            hstate_ego = initialize_ast_carry(eval_config, num_eval_envs)
            prev_action_ego = jnp.zeros(
                (num_eval_envs, env.action_space(env.agents[0]).n), dtype=jnp.float32
            )
            hstate_partner = ScannedRNN.initialize_carry(
                eval_config, rnn_type="gru", batch_size=num_eval_envs
            )
            done_batch = jnp.zeros((num_eval_envs,), dtype=bool)
            ep_return = jnp.zeros((num_eval_envs,), dtype=jnp.float32)

            def _step_fn(carry, _):
                obs, env_state, hstate_ego, hstate_partner, prev_action_ego, done_batch, ep_return, rng = carry

                rng, _rng_ego, _rng_partner, _rng_step = jax.random.split(rng, 4)

                obs_ego = obs[env.agents[0]]
                obs_partner = obs[env.agents[1]]

                ac_in_ego = (
                    obs_ego[jnp.newaxis, :],
                    done_batch[jnp.newaxis, :],
                    prev_action_ego[jnp.newaxis, :],
                )
                hstate_ego, pi_ego, _, _ = network_ego.apply(ego_params, hstate_ego, ac_in_ego)

                ac_in_partner = (obs_partner[jnp.newaxis, :], done_batch[jnp.newaxis, :])
                hstate_partner, pi_partner, _ = network_partner.apply(
                    partner_params, hstate_partner, ac_in_partner
                )

                action_ego = jnp.argmax(pi_ego.logits, axis=-1).squeeze(0)
                action_partner = jnp.argmax(pi_partner.logits, axis=-1).squeeze(0)

                next_prev_action_ego = jax.nn.one_hot(
                    action_ego, env.action_space(env.agents[0]).n
                )

                env_act = {
                    env.agents[0]: action_ego,
                    env.agents[1]: action_partner,
                }

                step_rng = jax.random.split(_rng_step, num_eval_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_rng, env_state, env_act)

                reward_ego = reward[env.agents[0]]
                new_done_batch = done["__all__"]
                ep_return = ep_return + reward_ego

                new_carry = (
                    next_obs,
                    next_env_state,
                    hstate_ego,
                    hstate_partner,
                    next_prev_action_ego,
                    new_done_batch,
                    ep_return,
                    rng,
                )
                return new_carry, None

            init_carry = (
                obsv,
                env_state,
                hstate_ego,
                hstate_partner,
                prev_action_ego,
                done_batch,
                ep_return,
                rng,
            )

            final_carry, _ = jax.lax.scan(_step_fn, init_carry, None, length=num_eval_steps)
            ep_return = final_carry[6]
            return ep_return.mean()

        rngs = jax.random.split(rng, num_eval_episodes)
        returns = jax.vmap(run_one_episode)(rngs)
        return {
            "zsc_eval/partner_mean_return": returns.mean(),
            "zsc_eval/partner_std_return": returns.std(),
        }

    def evaluator(rng, ego_params):
        num_partners = jax.tree_util.tree_leaves(eval_pop_params)[0].shape[0]
        partner_rngs = jax.random.split(rng, num_partners)

        def eval_partner_i(i, rng_i):
            partner_params = jax.tree_util.tree_map(lambda x: x[i], eval_pop_params)
            return evaluate_one_partner(rng_i, ego_params, partner_params)

        metrics = jax.vmap(eval_partner_i)(jnp.arange(num_partners), partner_rngs)

        mean_returns = metrics["zsc_eval/partner_mean_return"]
        std_returns = metrics["zsc_eval/partner_std_return"]

        return {
            "zsc_eval/mean_return": mean_returns.mean(),
            "zsc_eval/std_across_partners": mean_returns.std(),
            "zsc_eval/min_partner_return": mean_returns.min(),
            "zsc_eval/max_partner_return": mean_returns.max(),
            "zsc_eval/mean_intra_partner_std": std_returns.mean(),
        }

    return evaluator
    
def run_final_zsc_eval(config, final_params, eval_pop_params):
    evaluator = jax.jit(make_zsc_evaluator(config, eval_pop_params))
    
    num_seeds = jax.tree_util.tree_leaves(final_params)[0].shape[0]

    base_rng = jax.random.PRNGKey(config["SEED"] + 999)
    rngs = jax.random.split(base_rng, num_seeds)

    final_metrics = jax.vmap(evaluator, in_axes=(0, 0))(rngs, final_params)

    summary = jax.tree_util.tree_map(lambda x: x.mean(), final_metrics)
    summary_std = jax.tree_util.tree_map(lambda x: x.std(), final_metrics)

    print("\n===== FINAL HELD-OUT ZSC EVAL =====")
    print("Mean return:", summary["zsc_eval/mean_return"])
    print("Std across seeds:", summary_std["zsc_eval/mean_return"])
    print("Mean min partner return:", summary["zsc_eval/min_partner_return"])

    wandb.log({
        "final_zsc/mean_return": summary["zsc_eval/mean_return"],
        "final_zsc/std_across_seeds": summary_std["zsc_eval/mean_return"],
        "final_zsc/std_across_partners": summary["zsc_eval/std_across_partners"],
        "final_zsc/min_partner_return": summary["zsc_eval/min_partner_return"],
        "final_zsc/max_partner_return": summary["zsc_eval/max_partner_return"],
        "final_zsc/mean_intra_partner_std": summary["zsc_eval/mean_intra_partner_std"],
    })

    return final_metrics

def make_train(config, eval_pop_params):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    zsc_evaluator = make_zsc_evaluator(config, eval_pop_params)

    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = OvercookedV2LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = config["LR"]

        lr_warmup = config["LR_WARMUP"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = (
            config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        )

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        print("Update steps: ", update_steps)
        print("Warmup epochs: ", warmup_steps)
        print("Cosine epochs: ", cosine_epochs)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng, pop_params):

        # INIT NETWORK
        network_ego = ActorCriticASTRNN(
            env.action_space(env.agents[0]).n,
            config=config,
        )
        init_hstate_ego = initialize_ast_carry(config, config["NUM_ENVS"])     

        # Redefine network_partner: This is just the structure not the weights.
        network_partner = ActorCriticRNN(env.action_space(env.agents[1]).n, config=config, rnn_type="gru")  # Partner is always a baseline RNN
        

        rng, _rng_ego = jax.random.split(rng)

        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space().shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )

        network_params_ego = network_ego.init(_rng_ego, init_hstate_ego, init_x)

        # Create Train State ONLY for the Ego Agent
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state_ego = TrainState.create(
            apply_fn=network_ego.apply, params=network_params_ego, tx=tx
        )


        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        prev_action_ego = jnp.zeros(
            (config["NUM_ENVS"], env.action_space(env.agents[0]).n),
            dtype=jnp.float32
        )
        hstate_ego_start = initialize_ast_carry(config, config["NUM_ENVS"])

        hstate_partner_start = ScannedRNN.initialize_carry(
            config, rnn_type="gru", batch_size=config["NUM_ENVS"]
        )
        init_hstate_combined = (hstate_ego_start, hstate_partner_start)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate_combined,
                    prev_action_ego,
                    rng,
                ) = runner_state

                # Unpack the states
                hstate_ego, hstate_partner = hstate_combined
                
                rng, _rng_ego, _rng_partner, _rng_sample = jax.random.split(rng, 4)

                # --- EGO FORWARD PASS (Agent 0) ---
                obs_ego = last_obs[env.agents[0]]
                ac_in_ego = (
                    obs_ego[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                    prev_action_ego[jnp.newaxis, :],
                )

                

                # This naturally works for both ToM (tuple) and Baseline (array)
                hstate_ego, pi_ego, value_ego, aux_ego = network_ego.apply(
                    train_state.params, hstate_ego, ac_in_ego
                )
                action_ego = pi_ego.sample(seed=_rng_ego)
                log_prob_ego = pi_ego.log_prob(action_ego)

                next_prev_action_ego = jax.nn.one_hot(
                    action_ego.squeeze(0), env.action_space(env.agents[0]).n
                )

                # --- PARTNER FORWARD PASS (Agent 1) ---
                obs_partner = last_obs[env.agents[1]]
                ac_in_partner = (obs_partner[np.newaxis, :], last_done[np.newaxis, :])

                partner_idx = jax.random.randint(_rng_sample, (), 0, config["POP_SIZE"])
                sampled_partner_params = jax.tree_util.tree_map(lambda x: x[partner_idx], pop_params)

                hstate_partner, pi_partner, _ = network_partner.apply(
                    sampled_partner_params, hstate_partner, ac_in_partner
                )
                action_partner = pi_partner.sample(seed=_rng_partner)

                # COMBINE ACTIONS
                env_act = {
                    env.agents[0]: action_ego.flatten(),
                    env.agents[1]: action_partner.flatten()
                }

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                # In cooperative Overcooked, reward is shared, but we explicitly slice for Ego
                original_reward_ego = reward[env.agents[0]]

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                
                # Apply reward shaping
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                shaped_reward_ego = info["shaped_reward"][env.agents[0]]
                combined_reward_ego = reward[env.agents[0]]

                # Store metrics in info for WandB callback (Ego only)
                info["shaped_reward"] = shaped_reward_ego
                info["original_reward"] = original_reward_ego
                info["anneal_factor"] = jnp.full_like(shaped_reward_ego, anneal_factor)
                info["combined_reward"] = combined_reward_ego

                # We only need the 'done' flag for the environment
                done_batch = done["__all__"]

                # Create transition ONLY for Ego agent (Agent 0)
                transition = Transition(
                    done_batch,
                    action_ego.squeeze(),
                    value_ego.squeeze(),
                    combined_reward_ego.squeeze(),
                    log_prob_ego.squeeze(),
                    obs_ego,
                    prev_action_ego,
                    aux_ego["attended_feat"].squeeze(0),
                    aux_ego["pred_attended_feat"].squeeze(0),
                    aux_ego["gate_logits"].squeeze(0),
                    info,
                )
                # Pack the hidden states back together for the scan loop carry
                new_hstate_combined = (hstate_ego, hstate_partner)

                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    new_hstate_combined,
                    next_prev_action_ego,
                    rng,
                )
                return runner_state, transition

            hstate_combined = runner_state[-3]
            hstate_ego_init, _ = hstate_combined
            initial_hstate = hstate_ego_init
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, update_step, hstate, prev_action_ego, rng = (
                runner_state
            )
            hstate_ego, _ = hstate
            last_obs_ego = last_obs[env.agents[0]]
            ac_in = (
                last_obs_ego[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
                prev_action_ego[jnp.newaxis, :],
            )
            _, _, last_val, _ = network_ego.apply(train_state.params, hstate_ego, ac_in)


            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info
                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        squeezed_hstate = jax.tree_util.tree_map(
                            lambda x: jnp.squeeze(x, axis=0),
                            init_hstate,
                        )
                        _, pi, value, aux = network_ego.apply(
                            params,
                            squeezed_hstate,
                            (traj_batch.obs, traj_batch.done, traj_batch.prev_action),
                        )

                        # Attention Prediction Loss
                        pred_loss = jnp.square(
                            jax.lax.stop_gradient(aux["attended_feat"]) - aux["pred_attended_feat"]
                        ).mean()

                        gate_l2 = jnp.square(aux["gate_logits"]).mean()

                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config["AST_PRED_COEF"] * pred_loss
                            + config["AST_GATE_L2_COEF"] * gate_l2
                        )
                        return total_loss, (value_loss, loss_actor, entropy, pred_loss, gate_l2)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                init_hstate = jax.tree_util.tree_map(
                    lambda x: jnp.expand_dims(x, axis=0),
                    init_hstate,
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), init_hstate),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            total_loss_vals, aux_vals = loss_info
            value_loss_vals, actor_loss_vals, entropy_vals, pred_loss_vals, gate_l2_vals = aux_vals

            metric["loss/total_loss"] = total_loss_vals.mean()
            metric["loss/value_loss"] = value_loss_vals.mean()
            metric["loss/actor_loss"] = actor_loss_vals.mean()
            metric["loss/entropy"] = entropy_vals.mean()
            metric["ast/pred_loss"] = pred_loss_vals.mean()
            metric["ast/gate_l2"] = gate_l2_vals.mean()
            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            # periodic held-out ZSC evaluation
            rng, eval_rng = jax.random.split(rng)

            def _run_eval(args):
                params, rng = args
                return zsc_evaluator(rng, params)

            def _skip_eval(args):
                return {
                    "zsc_eval/mean_return": jnp.nan,
                    "zsc_eval/std_across_partners": jnp.nan,
                    "zsc_eval/min_partner_return": jnp.nan,
                    "zsc_eval/max_partner_return": jnp.nan,
                    "zsc_eval/mean_intra_partner_std": jnp.nan,
                }

            eval_metrics = jax.lax.cond(
                (update_step % config["EVAL_EVERY"] == 0),
                _run_eval,
                _skip_eval,
                operand=(train_state.params, eval_rng),
            )
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric = {**metric, **eval_metrics}
            jax.debug.callback(callback, metric)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
                prev_action_ego,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_ego,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            init_hstate_combined,
            prev_action_ego,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        final_train_state = runner_state[0]
        return {
            "runner_state": runner_state,
            "metrics": metric,
            "final_params": final_train_state.params,
        }

    return train


@hydra.main(
    version_base=None, config_path="config/oc_extended", config_name="fcp_overcooked_v2"
)
def main(config):
    config = OmegaConf.to_container(config)
    # overwrite the FCP checkpoints here for ease of use. Make sure to set CHECKPOINTS_PREFIX if your checkpoints are not in the default location.
    checkpoint_steps = [44,132,220]
    final_step = 220

    # Example: seeds 0-7 for FCP training pool, seeds 8-9 for held-out ZSC eval
    train_partner_idx = [i for i in range(5)]
    eval_partner_idx = [i for i in range(5,15)]

    config["FCP_TRAIN_CHECKPOINTS"] = []
    config["FCP_EVAL_CHECKPOINTS"] = []

    for p_idx in train_partner_idx:
        for ckpt_step in checkpoint_steps:
            config["FCP_TRAIN_CHECKPOINTS"].append(
                f"baseline_seed_{p_idx}_step_{ckpt_step}.msgpack"
            )

    for p_idx in eval_partner_idx:
        config["FCP_EVAL_CHECKPOINTS"].append(
            f"baseline_seed_{p_idx}_step_{final_step}.msgpack"
        )

    config["POP_SIZE"] = len(config["FCP_TRAIN_CHECKPOINTS"])
    config["EVAL_POP_SIZE"] = len(config["FCP_EVAL_CHECKPOINTS"])
    config["EVAL_EVERY"] = config.get("EVAL_EVERY", 50)
    config["EVAL_NUM_ENVS"] = config.get("EVAL_NUM_ENVS", 128)
    config["EVAL_NUM_EPISODES"] = config.get("EVAL_NUM_EPISODES", 8)
    config["EVAL_NUM_STEPS"] = config.get("EVAL_NUM_STEPS", config["NUM_STEPS"])

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    if config.get("AST_USE_OTHER_STREAM", True):
        model_name = f"ast"
    elif not(config.get("AST_USE_OTHER_STREAM", True)):
        model_name = f"ast_same_inp"

    print(f"Using {model_name}")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "OvercookedV2"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"{model_name}_fcp_overcooked_v2_{layout_name}",
    )

    # Initialize fixed Partner template
    env_temp = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network_partner = ActorCriticRNN(
        env_temp.action_space(env_temp.agents[1]).n, config=config, rnn_type="gru"
    )

    init_x = (
        jnp.zeros((1, config["NUM_ENVS"], *env_temp.observation_space().shape)),
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(config, rnn_type="gru")
    dummy_params = network_partner.init(jax.random.PRNGKey(0), init_hstate, init_x)

    train_pop_params = load_partner_pool(
        config, config["FCP_TRAIN_CHECKPOINTS"], dummy_params
    )
    eval_pop_params = load_partner_pool(
        config, config["FCP_EVAL_CHECKPOINTS"], dummy_params
    )

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config, eval_pop_params))
        out = jax.vmap(train_jit, in_axes=(0, None))(rngs, train_pop_params)

        final_metrics = run_final_zsc_eval(
            config=config,
            final_params=out["final_params"],
            eval_pop_params=eval_pop_params,
        )

if __name__ == "__main__":
    main()
