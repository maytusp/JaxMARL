# Fictitious Co-Play (FCP)
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
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
        
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(
            resets[:, np.newaxis],
            new_carry,
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


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
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

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


class ActorCriticToMRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        param_mode = self.config.get("PARAM_MODE", "hybrid")

        obs_self = obs
        obs_other = OvercookedToMTransform()(obs)

        if param_mode == "shared":
            hidden_self, hidden_other = hidden

            embed_model = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="cnn_shared",
            )
            layernorm = nn.LayerNorm(name="layernorm_shared")
            rnn = ScannedRNN(name="rnn_shared")

            emb_self = jax.vmap(embed_model)(obs_self)
            emb_other = jax.vmap(embed_model)(obs_other)

            emb_self = layernorm(emb_self)
            emb_other = layernorm(emb_other)

            hidden_self, emb_self = rnn(hidden_self, (emb_self, dones))
            hidden_other, emb_other = rnn(hidden_other, (emb_other, dones))

            if self.config.get("STOP_GRAD_OTHER", False):
                emb_other = jax.lax.stop_gradient(emb_other)
                hidden_other = jax.lax.stop_gradient(hidden_other)

            combined_embedding = jnp.concatenate([emb_self, emb_other], axis=-1)
            new_hidden = (hidden_self, hidden_other)

        elif param_mode == "shared_aggregate":
            # 1. hidden is now a single aggregated memory (h_agg_prev)
            hidden_agg = hidden

            embed_model = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="cnn_shared",
            )
            layernorm = nn.LayerNorm(name="layernorm_shared")
            rnn = ScannedRNN(name="rnn_shared")

            emb_self = jax.vmap(embed_model)(obs_self)
            emb_other = jax.vmap(embed_model)(obs_other)

            emb_self = layernorm(emb_self)
            emb_other = layernorm(emb_other)

            # 2. Both streams use the same aggregated memory (h_agg_prev)
            hidden_self, emb_self = rnn(hidden_agg, (emb_self, dones))
            hidden_other, emb_other = rnn(hidden_agg, (emb_other, dones))

            if self.config.get("STOP_GRAD_OTHER", False):
                emb_other = jax.lax.stop_gradient(emb_other)
                hidden_other = jax.lax.stop_gradient(hidden_other)

            # 3. Aggregate outputs for actor-critic instead of concatenation
            combined_embedding = emb_self + emb_other
            
            # 4. Aggregate memory for the next step (h_agg = h_self + h_other)
            new_hidden = hidden_self + hidden_other

        elif param_mode == "separate":
            hidden_self, hidden_other = hidden

            embed_model_self = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="cnn_self",
            )
            embed_model_other = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="cnn_other",
            )

            layernorm_self = nn.LayerNorm(name="layernorm_self")
            layernorm_other = nn.LayerNorm(name="layernorm_other")

            rnn_self = ScannedRNN(name="rnn_self")
            rnn_other = ScannedRNN(name="rnn_other")

            emb_self = jax.vmap(embed_model_self)(obs_self)
            if self.config.get("PERSPECTIVE_TRANSFORM", True):
                emb_other = jax.vmap(embed_model_other)(obs_other)
            else:
                emb_other = jax.vmap(embed_model_other)(obs_self)

            emb_self = layernorm_self(emb_self)
            emb_other = layernorm_other(emb_other)

            hidden_self, emb_self = rnn_self(hidden_self, (emb_self, dones))
            hidden_other, emb_other = rnn_other(hidden_other, (emb_other, dones))

            if self.config.get("STOP_GRAD_OTHER", False):
                emb_other = jax.lax.stop_gradient(emb_other)
                hidden_other = jax.lax.stop_gradient(hidden_other)

            combined_embedding = jnp.concatenate([emb_self, emb_other], axis=-1)
            new_hidden = (hidden_self, hidden_other)

        elif param_mode == "hybrid":
            hidden_self, hidden_other_shared, hidden_other_private = hidden

            # Shared branch
            embed_model_shared = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="cnn_shared",
            )
            layernorm_shared = nn.LayerNorm(name="layernorm_shared")
            rnn_shared = ScannedRNN(name="rnn_shared")

            # Private other branch
            embed_model_other_private = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="cnn_other_private",
            )
            layernorm_other_private = nn.LayerNorm(name="layernorm_other_private")
            rnn_other_private = ScannedRNN(name="rnn_other_private")

            # Self stream through shared params
            emb_self = jax.vmap(embed_model_shared)(obs_self)
            emb_self = layernorm_shared(emb_self)
            hidden_self, emb_self = rnn_shared(hidden_self, (emb_self, dones))

            # Other stream through shared params, but block gradient to shared params
            obs_other_sg = jax.lax.stop_gradient(obs_other)
            emb_other_shared = jax.vmap(embed_model_shared)(obs_other_sg)
            emb_other_shared = layernorm_shared(emb_other_shared)
            emb_other_shared = jax.lax.stop_gradient(emb_other_shared)
            hidden_other_shared, emb_other_shared = rnn_shared(
                hidden_other_shared, (emb_other_shared, dones)
            )
            emb_other_shared = jax.lax.stop_gradient(emb_other_shared)
            hidden_other_shared = jax.lax.stop_gradient(hidden_other_shared)

            # Other stream through private params
            emb_other_private = jax.vmap(embed_model_other_private)(obs_other)
            emb_other_private = layernorm_other_private(emb_other_private)
            hidden_other_private, emb_other_private = rnn_other_private(
                hidden_other_private, (emb_other_private, dones)
            )

            if self.config.get("STOP_GRAD_OTHER", False):
                emb_other_private = jax.lax.stop_gradient(emb_other_private)
                hidden_other_private = jax.lax.stop_gradient(hidden_other_private)

            combined_embedding = jnp.concatenate(
                [emb_self, emb_other_shared, emb_other_private], axis=-1
            )
            new_hidden = (hidden_self, hidden_other_shared, hidden_other_private)

        elif param_mode == "hybrid_ablate":
            hidden_self, hidden_other_a, hidden_other_b = hidden

            # Self stream (private)
            embed_model_self = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="cnn_self",
            )
            layernorm_self = nn.LayerNorm(name="layernorm_self")
            rnn_self = ScannedRNN(name="rnn_self")

            # Other stream A (private)
            embed_model_other_a = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="cnn_other_a",
            )
            layernorm_other_a = nn.LayerNorm(name="layernorm_other_a")
            rnn_other_a = ScannedRNN(name="rnn_other_a")

            # Other stream B (private)
            embed_model_other_b = CNN(
                output_size=self.config["GRU_HIDDEN_DIM"],
                activation=activation,
                name="cnn_other_b",
            )
            layernorm_other_b = nn.LayerNorm(name="layernorm_other_b")
            rnn_other_b = ScannedRNN(name="rnn_other_b")

            # Self stream
            emb_self = jax.vmap(embed_model_self)(obs_self)
            emb_self = layernorm_self(emb_self)
            hidden_self, emb_self = rnn_self(hidden_self, (emb_self, dones))

            # Other stream A
            emb_other_a = jax.vmap(embed_model_other_a)(obs_other)
            emb_other_a = layernorm_other_a(emb_other_a)
            hidden_other_a, emb_other_a = rnn_other_a(hidden_other_a, (emb_other_a, dones))

            # Other stream B
            emb_other_b = jax.vmap(embed_model_other_b)(obs_other)
            emb_other_b = layernorm_other_b(emb_other_b)
            hidden_other_b, emb_other_b = rnn_other_b(hidden_other_b, (emb_other_b, dones))

            if self.config.get("STOP_GRAD_OTHER", False):
                emb_other_a = jax.lax.stop_gradient(emb_other_a)
                hidden_other_a = jax.lax.stop_gradient(hidden_other_a)
                emb_other_b = jax.lax.stop_gradient(emb_other_b)
                hidden_other_b = jax.lax.stop_gradient(hidden_other_b)

            combined_embedding = jnp.concatenate(
                [emb_self, emb_other_a, emb_other_b], axis=-1
            )
            new_hidden = (hidden_self, hidden_other_a, hidden_other_b)

        else:
            raise ValueError(f"Unknown PARAM_MODE: {param_mode}")

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(combined_embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(combined_embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return new_hidden, pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

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
        use_tom = config.get("USE_TOM", True)
        param_mode = config.get("PARAM_MODE", "separate")
        # Redefine network_partner: This is just the structure not the weights.
        network_partner = ActorCriticRNN(env.action_space(env.agents[1]).n, config=config)
        
        if use_tom:
            print("Initializing Like-Me ToM Network")
            network_ego = ActorCriticToMRNN(env.action_space(env.agents[0]).n, config=config)

            if param_mode in ["shared", "separate"]:
                print(f"USE {'SHARED' if param_mode == 'shared' else 'SEPARATE'} PARAMS: Separate CNN, LayerNorm, and RNNs")
                init_hstate_ego = (
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
                )
            elif param_mode in ["hybrid", "hybrid_ablate"]:
                print("USE HYBRID PARAMS")
                init_hstate_ego = (
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),  # self
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),  # other shared
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),  # other private
                )
            elif param_mode in ["shared_aggregate"]:
                print("USE SHARED PARAMS WITH AGGREGATION")
                init_hstate_ego = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
            else:
                raise ValueError(f"Unknown PARAM_MODE: {param_mode}")
        else:
            print("Initializing Baseline RNN Network without ToM (No Shared Parameters)")
            network_ego = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
            init_hstate_ego = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        rng, _rng_ego = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space().shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
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

        # Create fresh hidden states for the start of the rollout
        if config.get("USE_TOM", True):
            param_mode = config.get("PARAM_MODE", "separate")
            if param_mode in ["shared", "separate"]:
                hstate_ego_start = (
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
                )
            elif param_mode in ["hybrid", "hybrid_ablate"]:
                hstate_ego_start = (
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),  # self
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),  # other shared
                    ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),  # other private
                )
            elif param_mode in ["shared_aggregate"]:
                hstate_ego_start = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        else:
            hstate_ego_start = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
            
        hstate_partner_start = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        # Combine them for the scan carry
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
                    rng,
                ) = runner_state

                # Unpack the states
                hstate_ego, hstate_partner = hstate_combined
                
                rng, _rng_ego, _rng_partner, _rng_sample = jax.random.split(rng, 4)

                # --- EGO FORWARD PASS (Agent 0) ---
                obs_ego = last_obs[env.agents[0]]
                ac_in_ego = (obs_ego[np.newaxis, :], last_done[np.newaxis, :])
                
                # This naturally works for both ToM (tuple) and Baseline (array)
                hstate_ego, pi_ego, value_ego = network_ego.apply(
                    train_state.params, hstate_ego, ac_in_ego
                )
                action_ego = pi_ego.sample(seed=_rng_ego)
                log_prob_ego = pi_ego.log_prob(action_ego)

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
                    rng,
                )
                return runner_state, transition

            hstate_combined = runner_state[-2]
            hstate_ego_init, _ = hstate_combined
            initial_hstate = hstate_ego_init
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
            )
            hstate_ego, _ = hstate
            last_obs_ego = last_obs[env.agents[0]]

            ac_in = (
                last_obs_ego[jnp.newaxis, :],   # (1, NUM_ENVS, H, W, C)
                last_done[jnp.newaxis, :],      # (1, NUM_ENVS)
            )

            _, _, last_val = network_ego.apply(train_state.params, hstate_ego, ac_in)
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
                        squeezed_hstate = jax.tree_util.tree_map(lambda x: x.squeeze(), init_hstate)
                        _, pi, value = network_ego.apply(
                            params,
                            squeezed_hstate,
                            (traj_batch.obs, traj_batch.done),
                        )

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
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

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
                            lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), 
                            init_hstate
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
                    jax.tree_util.tree_map(lambda x: x.squeeze(), init_hstate),
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

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
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
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(
    version_base=None, config_path="config", config_name="fcp_overcooked_v2"
)
def main(config):
    config = OmegaConf.to_container(config)

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    if config.get("USE_TOM", True):
        param_mode = config.get("PARAM_MODE", "hybrid")
        if param_mode == "shared":
            model_name = "lmtom"
        elif param_mode == "shared_aggregate":
            model_name = "lmtom_shared_agg"
        elif param_mode == "hybrid":
            model_name = "lmtom_hybrid"
        elif param_mode == "hybrid_ablate":
            model_name = "lmtom_hybrid_ablate"
        elif param_mode == "separate":
            model_name = "lmtom_ablate_share"
            
        if not(config.get("PERSPECTIVE_TRANSFORM", True)):
            model_name += "_same_input"

        if config.get("STOP_GRAD_OTHER"):
            model_name += "_stopgrad"
    else:
        model_name = "rnn_baseline"
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
    network_partner = ActorCriticRNN(env_temp.action_space(env_temp.agents[1]).n, config=config)
    
    # Same init_x as Phase 1
    init_x = (
        jnp.zeros((1, config["NUM_ENVS"], *env_temp.observation_space().shape)),
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    
    dummy_params = network_partner.init(jax.random.PRNGKey(0), init_hstate, init_x)
    # print("fresh dummy kernel shape:",
    #   dummy_params["params"]["CNN_0"]["Conv_0"]["kernel"].shape)
    # --- 2. LOAD PARTNERS ---
    checkpoints_prefix = config.get("CHECKPOINTS_PREFIX", "./checkpoints/fcp_pools")
    pool_dir = os.path.join(checkpoints_prefix, layout_name)
    
    loaded_params = []
    for ckpt_name in config["FCP_CHECKPOINTS"]:
        ckpt_path = os.path.join(pool_dir, ckpt_name)
        # print(f"ATTEMPTING TO LOAD: {ckpt_name}")
        # print(f"Loading fresh sanity check: {ckpt_name}")
        try:
            with open(ckpt_path, "rb") as f:
                # No hacks needed. It just fits perfectly.
                p = flax.serialization.from_bytes(dummy_params, f.read())
                print("loaded kernel shape:",
                p["params"]["CNN_0"]["Conv_0"]["kernel"].shape)
                loaded_params.append(p)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find FCP checkpoint at {ckpt_path}.")

    # Stack them into shape [POP_SIZE, ...]
    pop_params = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *loaded_params)

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit, in_axes=(0, None))(rngs, pop_params)

if __name__ == "__main__":
    main()
