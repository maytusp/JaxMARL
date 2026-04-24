"""
Conceptual Perspective Taking
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

"""
import os
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.core import freeze, unfreeze
from flax.core.frozen_dict import FrozenDict
from flax import traverse_util
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9

import wandb
import functools
import pdb
from jax_tqdm import scan_tqdm


def compute_checkpoint_boundaries(total_updates, num_checkpoints=10):
    boundaries = []
    for checkpoint_idx in range(1, num_checkpoints + 1):
        boundary = int(np.ceil(total_updates * checkpoint_idx / num_checkpoints))
        boundary = min(total_updates, boundary)
        if boundary > 0 and (not boundaries or boundary != boundaries[-1]):
            boundaries.append(boundary)
    return boundaries


def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config['layout_name'] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["ENV_NAME"] == "overcooked":
        def reset_env(key):
            def reset_sub_dict(key, fn):
                key, subkey = jax.random.split(key)
                sampled_layout_dict = fn(subkey, ik=True)
                temp_o, temp_s = env.custom_reset(key, layout=sampled_layout_dict, random_reset=False, shuffle_inv_and_pot=False)
                key, subkey = jax.random.split(key)
                return (temp_o, temp_s), key
                
            asymm_reset, key = reset_sub_dict(key, make_asymm_advantages_9x9)
            coord_ring_reset, key = reset_sub_dict(key, make_coord_ring_9x9)
            counter_circuit_reset, key = reset_sub_dict(key, make_counter_circuit_9x9)
            forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
            cramped_room_reset, key = reset_sub_dict(key, make_cramped_room_9x9)
            layout_resets = [asymm_reset, coord_ring_reset, counter_circuit_reset, forced_coord_reset, cramped_room_reset]
            # stack all layouts
            stacked_layout_reset = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *layout_resets)
            # sample an index from 0 to 4
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset = jax.tree_util.tree_map(lambda x: x[index], stacked_layout_reset)
            return sampled_reset
        @scan_tqdm(100)
        def gen_held_out(runner_state, unused):
            (i,) = runner_state
            _, ho_state = reset_env(jax.random.key(i))
            res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
            carry = (i+1,)
            return carry, res
        carry, res = jax.lax.scan(gen_held_out, (0,), jnp.arange(100), 100)
        ho_goal, ho_wall, ho_pot = [], [], []
        for layout_name, layout_dict in overcooked_layouts.items():  # add hand crafted ones to heldout set
            if "9" in layout_name:
                _, ho_state = env.custom_reset(jax.random.PRNGKey(0), random_reset=False, shuffle_inv_and_pot=False, layout=layout_dict)
                ho_goal.append(ho_state.goal_pos)
                ho_wall.append(ho_state.wall_map)
                ho_pot.append(ho_state.pot_pos)
        ho_goal = jnp.stack(ho_goal, axis=0)
        ho_wall = jnp.stack(ho_wall, axis=0)
        ho_pot = jnp.stack(ho_pot, axis=0)
        ho_goal = jnp.concatenate([res[0], ho_goal], axis=0)
        ho_wall = jnp.concatenate([res[1], ho_wall], axis=0)
        ho_pot = jnp.concatenate([res[2], ho_pot], axis=0)
        env.held_out_goal, env.held_out_wall, env.held_out_pot = (ho_goal, ho_wall, ho_pot)
    elif config["ENV_NAME"] == "ToyCoop":
        # Generate 100 held-out states for ToyCoop
        @scan_tqdm(100)
        def gen_held_out_toycoop(runner_state, unused):
            (i,) = runner_state
            key = jax.random.key(i)
            state = env.custom_reset_fn(key, random_reset=True)
            res = (state.agent_pos, state.goal_pos, state.other_goal_pos)
            carry = (i+1,)
            return carry, res
        
        carry, res = jax.lax.scan(gen_held_out_toycoop, (0,), jnp.arange(100), 100)
        ho_agent_pos, ho_goal_pos, ho_other_goal_pos = res
        
        # Set the held-out states in the environment
        env.held_out_agent_pos = ho_agent_pos
        env.held_out_goal_pos = ho_goal_pos
        env.held_out_other_goal_pos = ho_other_goal_pos
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env

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
        lstm_state = carry
        ins, resets = x
        
        # Reset LSTM state on episode boundaries
        lstm_state = jax.tree_util.tree_map(
            lambda x: jnp.where(resets[:, np.newaxis], jnp.zeros_like(x), x),
            lstm_state
        )
        
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[-1])(lstm_state, ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.OptimizedLSTMCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )



class PhaseOneActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, agent_positions = x
        batch_size, num_envs, flattened_obs_dim = obs.shape
        if self.config["GRAPH_NET"]:
            if self.config["ENV_NAME"] == "overcooked":
                reshaped_obs = obs.reshape(-1, 7, 7, 26)
            else:
                reshaped_obs = obs.reshape(-1, 5, 5, 4)

            embedding = nn.Conv(
                features=64,
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
                features=32,
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)
            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = nn.relu(embedding)

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = nn.relu(embedding)

        if self.config["LSTM"]:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)
        else:
            embedding = nn.Dense(
                self.config["GRU_HIDDEN_DIM"],
                kernel_init=orthogonal(2),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)
        embedding = embedding.reshape((batch_size, num_envs, -1))

        actor_hidden1 = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_hidden1 = nn.relu(actor_hidden1)
        actor_hidden2 = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] * 3 // 4,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_hidden1)
        actor_hidden2 = nn.relu(actor_hidden2)
        actor_hidden3 = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_hidden2)
        actor_hidden3 = nn.relu(actor_hidden3)
        if self.config["ENV_NAME"] == "overcooked":
            actor_hidden4 = nn.Dense(
                self.config["GRU_HIDDEN_DIM"] // 4,
                kernel_init=orthogonal(2),
                bias_init=constant(0.0),
            )(actor_hidden3)
            actor_hidden4 = nn.relu(actor_hidden4)
        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_hidden4)

        critic_hidden1 = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic_hidden1 = nn.relu(critic_hidden1)
        critic_hidden2 = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic_hidden1)
        critic_hidden2 = nn.relu(critic_hidden2)
        if self.config["ENV_NAME"] == "overcooked":
            critic_hidden3 = nn.Dense(
                self.config["FC_DIM_SIZE"] * 3 // 4,
                kernel_init=orthogonal(2),
                bias_init=constant(0.0),
            )(critic_hidden2)
            critic_hidden3 = nn.relu(critic_hidden3)
            critic_hidden4 = nn.Dense(
                self.config["FC_DIM_SIZE"] // 2,
                kernel_init=orthogonal(2),
                bias_init=constant(0.0),
            )(critic_hidden3)
            critic_hidden4 = nn.relu(critic_hidden4)
        critic_value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic_hidden4)

        return hidden, actor_hidden4, actor_logits, critic_hidden4, jnp.squeeze(critic_value, axis=-1)


class TransformerFusionBlock(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        norm_x = nn.LayerNorm()(x)
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(norm_x)
        x = x + attn_out

        mlp_in = nn.LayerNorm()(x)
        mlp_out = nn.Dense(
            self.embed_dim * 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(mlp_in)
        mlp_out = nn.relu(mlp_out)
        mlp_out = nn.Dense(
            self.embed_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(mlp_out)
        return x + mlp_out


class ActorCriticCPT(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, partner_obs, dones, agent_positions = x
        self_hidden, other_hidden = hidden

        self_hidden, self_actor_features, _, self_critic_features, _ = PhaseOneActorCritic(
            self.action_dim, self.config, name="self_stream"
        )(
            self_hidden, (obs, dones, agent_positions)
        )
        other_hidden, other_actor_features, _, other_critic_features, _ = PhaseOneActorCritic(
            self.action_dim, self.config, name="other_stream"
        )(
            other_hidden, (partner_obs, dones, agent_positions)
        )

        if self.config.get("CPT_FREEZE_SELF_STREAM", True):
            self_actor_features = jax.lax.stop_gradient(self_actor_features)
            self_critic_features = jax.lax.stop_gradient(self_critic_features)
        if self.config.get("CPT_FREEZE_OTHER_STREAM", True):
            other_actor_features = jax.lax.stop_gradient(other_actor_features)
            other_critic_features = jax.lax.stop_gradient(other_critic_features)

        actor_feature_dim = self.config["GRU_HIDDEN_DIM"] // 4 if self.config["ENV_NAME"] == "overcooked" else self.config["GRU_HIDDEN_DIM"] // 2
        critic_feature_dim = self.config["FC_DIM_SIZE"] // 2 if self.config["ENV_NAME"] == "overcooked" else self.config["FC_DIM_SIZE"]
        fusion_dim = max(actor_feature_dim, critic_feature_dim)

        def pad_features(features, target_dim):
            pad_width = target_dim - features.shape[-1]
            if pad_width <= 0:
                return features
            padding = [(0, 0)] * features.ndim
            padding[-1] = (0, pad_width)
            return jnp.pad(features, padding)

        tokens = jnp.stack(
            [
                pad_features(self_actor_features, fusion_dim),
                pad_features(other_actor_features, fusion_dim),
                pad_features(self_critic_features, fusion_dim),
                pad_features(other_critic_features, fusion_dim),
            ],
            axis=-2,
        )
        num_heads = self.config.get("CPT_NUM_HEADS", 4)
        if fusion_dim % num_heads != 0:
            num_heads = 1
        fused_tokens = TransformerFusionBlock(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            name="fusion_block",
        )(tokens)

        fused_actor_features = fused_tokens[..., :2, :].mean(axis=-2)[..., :actor_feature_dim]
        fused_critic_features = fused_tokens[..., 2:, :].mean(axis=-2)[..., :critic_feature_dim]

        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )(fused_actor_features)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(fused_critic_features)

        pi = distrax.Categorical(logits=actor_logits)
        return (self_hidden, other_hidden), pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    partner_obs: jnp.ndarray
    info: jnp.ndarray
    agent_positions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def extract_param_subtree(variables):
    if variables is None:
        return None
    if isinstance(variables, dict) or hasattr(variables, "keys"):
        if "params" in variables:
            return variables["params"]
    return variables


def build_trainable_labels(params, trainable_top_level_paths):
    flat_params = traverse_util.flatten_dict(unfreeze(params))
    flat_labels = {}
    for key in flat_params.keys():
        is_trainable = any(key[: len(path)] == path for path in trainable_top_level_paths)
        flat_labels[key] = "train" if is_trainable else "freeze"
    return freeze(traverse_util.unflatten_dict(flat_labels))


def extract_dense_layer_params_by_output_dim(param_tree, output_dim):
    flat_params = traverse_util.flatten_dict(unfreeze(param_tree))
    matching_prefixes = []
    for key, value in flat_params.items():
        if key[-1] != "kernel" or value.ndim != 2 or value.shape[-1] != output_dim:
            continue
        bias_key = key[:-1] + ("bias",)
        if bias_key in flat_params:
            matching_prefixes.append(key[:-1])

    if not matching_prefixes:
        raise ValueError(f"Could not find dense layer with output dim {output_dim}")

    selected_prefix = max(matching_prefixes)
    return freeze(
        {
            "kernel": flat_params[selected_prefix + ("kernel",)],
            "bias": flat_params[selected_prefix + ("bias",)],
        }
    )


def resolve_phase_one_ckpt_path(config, fcp_prefix, suffix):
    explicit_path = config["TRAIN_KWARGS"].get("phase1_ckpt_path")
    if explicit_path:
        return explicit_path

    phase1_layout = config["TRAIN_KWARGS"].get(
        "phase1_layout",
        config["ENV_KWARGS"].get("layout", "cramped_room_9"),
    )
    phase1_random_reset = config["TRAIN_KWARGS"].get(
        "phase1_random_reset",
        True,
    )
    phase1_reset_fn = config["TRAIN_KWARGS"].get(
        "phase1_random_reset_fn",
        config["ENV_KWARGS"].get("random_reset_fn", "reset_all"),
    )
    phase1_graph = config["TRAIN_KWARGS"].get("phase1_graph_net", config["GRAPH_NET"])
    phase1_seed = config["TRAIN_KWARGS"].get("phase1_seed", config["SEED"])
    phase1_ckpt_id = config["TRAIN_KWARGS"].get("phase1_ckpt_id", 0)

    phase1_filepath = f"ckpts/ippo/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        phase1_filepath += f"/{phase1_layout}"
    phase1_filepath = (
        f"{phase1_filepath}/ik{phase1_random_reset}/{phase1_reset_fn}/graph{phase1_graph}"
    )
    return f"{phase1_filepath}/{fcp_prefix}seed{phase1_seed}_ckpt{phase1_ckpt_id}{suffix}.pkl"


def make_train(config, update_step=0):
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = initialize_environment(config)
    
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    resume_update_step = update_step * (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
    config["MAX_TRAIN_UPDATES"] = (
        config["MAX_TRAIN_STEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_REWARD_SHAPING_STEPS"] = config["MAX_TRAIN_UPDATES"] // 2  # used for annealing reward shaping
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )
    config["obs_dim"] = env.observation_space(env.agents[0]).shape

    obs, state = env.reset(jax.random.PRNGKey(0), params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    def linear_schedule(count):
        frac = (
            1.0
            - ((count + resume_update_step) // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["MAX_TRAIN_UPDATES"]
        )
        frac = jnp.maximum(1e-9, frac)
        return config["LR"] * frac

    def make_partner_obs(obs_batch):
        if config.get("CPT_CONTROL_SELF_ONLY", False):
            return obs_batch
        if config["ENV_NAME"] != "overcooked":
            return obs_batch
        obs_image = obs_batch.reshape((-1, *config["obs_dim"]))
        partner_obs = jax.vmap(env.perspective_transform)(obs_image)
        return partner_obs.reshape((obs_batch.shape[0], -1))

    checkpoint_boundaries = compute_checkpoint_boundaries(int(config["NUM_UPDATES"]))
    checkpoint_boundary_to_id = {
        boundary: config["TRAIN_KWARGS"]["ckpt_id"] + idx
        for idx, boundary in enumerate(checkpoint_boundaries)
    }

    def train(rng, model_params=None, update_step=0, phase_one_params=None):
        # INIT NETWORK
        network = ActorCriticCPT(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        # get flattened obs dim
        flattened_obs_dim = 1
        for dim in env.observation_space(env.agents[0]).shape:
            flattened_obs_dim *= dim
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], flattened_obs_dim)
            ),
            jnp.zeros(
                (1, config["NUM_ENVS"], flattened_obs_dim)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], 2, 2)).astype(jnp.int32)
        )
        init_hstate = (
            ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
            ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
        )
        network_params = network.init(_rng, init_hstate, init_x)
        if model_params is not None:
            network_params = model_params
        elif phase_one_params is not None:
            mutable_params = unfreeze(network_params)
            phase_one_subtree = extract_param_subtree(phase_one_params)
            mutable_params["params"]["self_stream"] = phase_one_subtree
            mutable_params["params"]["other_stream"] = phase_one_subtree
            mutable_params["params"]["actor_out"] = extract_dense_layer_params_by_output_dim(
                phase_one_subtree, env.action_space(env.agents[0]).n
            )
            mutable_params["params"]["critic_out"] = extract_dense_layer_params_by_output_dim(
                phase_one_subtree, 1
            )
            network_params = freeze(mutable_params)
        # Define all the parameter paths you want to actually update
        trainable_paths = {
            ("params", "fusion_block"),
            ("params", "self_stream"),
            ("params", "other_stream"),
            ("params", "actor_out"),
            ("params", "critic_out"),
        }

        # This builds the 'train' vs 'freeze' map for the optimizer
        trainable_labels = build_trainable_labels(network_params, trainable_paths)
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
        )
        tx = optax.multi_transform(
            {
                "train": base_optimizer,
                "freeze": optax.set_to_zero(),
            },
            trainable_labels,
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = (
            ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]),
            ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]),
        )

        # TRAIN LOOP
        @scan_tqdm(int(config["NUM_UPDATES"]))
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                partner_obs_batch = make_partner_obs(obs_batch)
                agent_positions = {'agent_0': env_state.env_state.agent_pos, 'agent_1': env_state.env_state.agent_pos}  
                agent_positions = batchify(agent_positions, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    partner_obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    agent_positions[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                shaped_reward = info['shaped_reward']
                reward_shaping_frac = jnp.maximum(0.0, 1.0 - (update_step / config["NUM_REWARD_SHAPING_STEPS"]))
                reward = jax.tree_util.tree_map(lambda x, y: x + y * reward_shaping_frac, reward, shaped_reward)
                
                # remove shaped rewards
                del info['shaped_reward']

                info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    partner_obs_batch,
                    info,
                    agent_positions
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_step)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            (train_state, env_state, obsv, done_batch, hstate, rng) = runner_state
            runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_steps)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng, update_steps = runner_state
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            partner_obs_batch = make_partner_obs(last_obs_batch)
            agent_positions = {'agent_0': env_state.env_state.agent_pos, 'agent_1': env_state.env_state.agent_pos}
            agent_positions = batchify(agent_positions, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                partner_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                agent_positions[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
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
                        _, pi, value = network.apply(
                            params,
                            jax.tree_util.tree_map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.partner_obs, traj_batch.done, traj_batch.agent_positions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
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

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jax.tree_util.tree_map(lambda h: jnp.reshape(h, (1, config["NUM_ACTORS"], -1)), init_hstate)
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
                    jax.tree_util.tree_map(lambda h: h.squeeze(), init_hstate),
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
            metric = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        # the metrics have an agent dimension, but this is identical
                        # for all agents so index into the 0th item of that dimension.
                        "returns": metric["returns"],
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        **metric["loss"],
                    }
                )

            def save_checkpoint(local_step, global_step, params, rng_key, returns):
                current_local_step = int(local_step)
                ckpt_id = checkpoint_boundary_to_id.get(current_local_step)
                if ckpt_id is None:
                    return

                save_path = (
                    f"{config['CHECKPOINT_DIR']}/{config['CHECKPOINT_PREFIX']}"
                    f"{ckpt_id}{config['CHECKPOINT_SUFFIX']}.pkl"
                )
                if (not config["TRAIN_KWARGS"]["overwrite_ckpt"]) and os.path.exists(save_path):
                    print(f"Checkpoint {ckpt_id} already exists, skipping save")
                    return

                host_params = jax.tree_util.tree_map(np.array, params)
                host_rng = np.array(rng_key)
                reward_value = float(np.array(returns))
                ckpt = {
                    "key": host_rng,
                    "params": host_params,
                    "final_update_step": int(global_step),
                    "first_update_step": int(global_step),
                    "last_update_step": int(global_step) * config["NUM_ENVS"] * config["NUM_STEPS"],
                    "first_reward": reward_value,
                    "middle_reward": reward_value,
                    "last_reward": reward_value,
                }
                with open(save_path, "wb") as f:
                    pickle.dump(ckpt, f)
                print(f"Saved checkpoint {ckpt_id} at update step {int(global_step)} to {save_path}")

            returns = metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0].astype(jnp.int32)
                        ].mean()
            metric["returns"] = returns
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            local_completed_update_step = update_steps - update_step + 1
            global_completed_update_step = update_steps + 1
            jax.debug.callback(
                save_checkpoint,
                local_completed_update_step,
                global_completed_update_step,
                train_state.params,
                rng,
                returns,
            )
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)  # hstate resets automatically
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, update_step), jnp.arange(int(config["NUM_UPDATES"])), int(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, 'metrics': metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="")
def main(config):
    config = OmegaConf.to_container(config)
    config.setdefault("CPT_NUM_HEADS", 4)
    config.setdefault("CPT_CONTROL_SELF_ONLY", False)
    config.setdefault("CPT_FREEZE_SELF_STREAM", False)
    config.setdefault("CPT_FREEZE_OTHER_STREAM", False)
    if config['TRAIN_KWARGS']['finetune']:
        config['LR'] = config['LR'] / 10
        finetune_appendage = "_improved_finetune"
        if config['FCP']:
            fcp_prefix = "fcp_"
        else:
            fcp_prefix = ""
    elif config['ENV_NAME'] == 'overcooked':
        fcp_prefix = ""
        finetune_appendage = "_improved"
    else:
        fcp_prefix = ""
        finetune_appendage = "_improved"
    
    if config['ENV_KWARGS']['partial_obs']:
        finetune_appendage += "_partial_obs"
    if not config['LSTM']:
        finetune_appendage += "_no_lstm"

    if config['ENV_NAME'] == 'ToyCoop':
        if config['ENV_KWARGS']['incentivize_strat'] != 2:
            finetune_appendage += f"_incentivize_strat_{config['ENV_KWARGS']['incentivize_strat']}"
    phase_one_suffix = "_improved"
    if config['ENV_KWARGS']['partial_obs']:
        phase_one_suffix += "_partial_obs"
    if not config['LSTM']:
        phase_one_suffix += "_no_lstm"
    
    if config['CPT_CONTROL_SELF_ONLY']:
        finetune_appendage += "_cpt_sameobs"
    elif not(config['CPT_CONTROL_SELF_ONLY']):
        finetune_appendage += "_cpt_nomasking"

    if not(config['CPT_FREEZE_SELF_STREAM'] and config['CPT_FREEZE_OTHER_STREAM']):
        finetune_appendage += "_ft"

    wandb.init(
        entity=config["ENTITY"],
        name=finetune_appendage,
        project=config["PROJECT"],
        tags=["IPPO", "CPT", "Transformer", "SP"],
        config=config,
        mode=config["WANDB_MODE"]
    )
    filepath = f"ckpts/ippo_cpt/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath += f"/{config['ENV_KWARGS']['layout']}"
    filepath = f'{filepath}/ik{config["ENV_KWARGS"]["random_reset"]}/{config["ENV_KWARGS"]["random_reset_fn"]}/graph{config["GRAPH_NET"]}'
    os.makedirs(filepath, exist_ok=True)
    config["CHECKPOINT_DIR"] = filepath
    config["CHECKPOINT_PREFIX"] = f"{fcp_prefix}seed{config['SEED']}_ckpt"
    config["CHECKPOINT_SUFFIX"] = finetune_appendage
    print(f"Working on: \n{filepath}\n")

    if not config['TRAIN_KWARGS']['overwrite_ckpt']:
        checkpoint_boundaries = compute_checkpoint_boundaries(
            int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        )
        for ckpt_offset in range(len(checkpoint_boundaries)):
            ckpt_id = config['TRAIN_KWARGS']['ckpt_id'] + ckpt_offset
            ckpt_path = f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{ckpt_id}{finetune_appendage}.pkl"
            if os.path.exists(ckpt_path):
                print(f"Checkpoint {ckpt_id} already exists, exiting")
                exit(0)

    if config['TRAIN_KWARGS']['ckpt_id'] > 0:
        print("Loading checkpoint")
        with open(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id'] - 1}{finetune_appendage}.pkl", "rb") as f:
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
            final_update_step = previous_ckpt['final_update_step']
            phase_one_params = None
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))

    else:
        phase_one_ckpt_path = resolve_phase_one_ckpt_path(config, fcp_prefix, phase_one_suffix)
        print(f"Loading phase-1 checkpoint for actor-critic initialization: {phase_one_ckpt_path}")
        with open(phase_one_ckpt_path, "rb") as f:
            previous_ckpt = pickle.load(f)
            model_params = None
            phase_one_params = previous_ckpt['params']
            final_update_step = 0
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))

    print(f"Starting from update step {final_update_step}")
    train_jit = jax.jit(make_train(config, final_update_step), device=jax.devices()[0])
    out = train_jit(rng, model_params, final_update_step, phase_one_params)
    runner_state = out['runner_state']
    train_state = runner_state[0]
    model_state = train_state[0]
    rng = runner_state[-1]
    metrics = out['metrics']

    reward = metrics['returns']
    update_step = metrics['update_steps']
    loss = metrics['loss']
    value_loss = loss['value_loss']
    actor_loss = loss['actor_loss']
    entropy_loss = loss['entropy']
    final_update_step = update_step[-1]
    update_step = update_step * config['NUM_ENVS'] * config['NUM_STEPS']


    # plot reward w wandb
    for i, us in enumerate(update_step):
        r = reward[i]
        try:
            wandb.log(
                {
                    "returns": r,
                    "env_step": us,
                    'seed': config["SEED"]
                }
            )
        except:
            pass




    # plot reward vs update step with seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_context('paper')
    # add previous ckpt's first and last update step and reward
    if config['TRAIN_KWARGS']['ckpt_id'] > 0:
        # plot_update_step = jnp.concatenate([, previous_ckpt['last_update_step'][None], update_step])    
        # plot_reward = jnp.concatenate([previous_ckpt['first_reward'][None], previous_ckpt['last_reward'][None], reward])
        plot_update_step = update_step
        plot_reward = reward
    else:
        plot_update_step = update_step
        plot_reward = reward

    value_step = jnp.arange(value_loss.shape[0])
    

    # plot all losses in subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # Changed to 3x2 to add ratio plot
    fig.suptitle('Training Losses')
    
    # Plot total loss
    sns.lineplot(x=value_step, y=loss['total_loss'], ax=axs[0, 0])
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].set_xlabel('Steps')
    
    # Plot value loss
    sns.lineplot(x=value_step, y=value_loss, ax=axs[0, 1])
    axs[0, 1].set_title('Value Loss')
    axs[0, 1].set_xlabel('Steps')
    
    # Plot actor loss
    sns.lineplot(x=value_step, y=loss['actor_loss'], ax=axs[1, 0])
    axs[1, 0].set_title('Actor Loss')
    axs[1, 0].set_xlabel('Steps')
    
    # Plot entropy loss
    sns.lineplot(x=value_step, y=entropy_loss, ax=axs[1, 1])
    axs[1, 1].set_title('Entropy Loss')
    axs[1, 1].set_xlabel('Steps')
    
    # Plot ratio
    sns.lineplot(x=value_step, y=loss['ratio'], ax=axs[2, 0])
    axs[2, 0].set_title('Policy Ratio')
    axs[2, 0].set_xlabel('Steps')
    
    # Hide the empty subplot
    sns.lineplot(x=plot_update_step, y=plot_reward, ax=axs[2, 1])
    axs[2, 1].set_title('Reward')
    axs[2, 1].set_xlabel('Steps')
    
    final_ckpt_id = config["TRAIN_KWARGS"]["ckpt_id"] + max(
        len(compute_checkpoint_boundaries(int(config["NUM_UPDATES"]))) - 1,
        0,
    )
    plt.tight_layout()
    plt.savefig(f'{filepath}/{fcp_prefix}train_info_seed{config["SEED"]}_ckpt{final_ckpt_id}{finetune_appendage}.png')
    plt.close()

    print(f"Finished training for seed {config['SEED']} with final ckpt {final_ckpt_id}")
    print(f'Saved to {filepath}/{fcp_prefix}train_info_seed{config["SEED"]}_ckpt{final_ckpt_id}{finetune_appendage}.png')
    
    
    '''updates_x = jnp.arange(out["metrics"]["total_loss"][0].shape[0])
    loss_table = jnp.stack([updates_x, out["metrics"]["total_loss"].mean(axis=0), out["metrics"]["actor_loss"].mean(axis=0), out["metrics"]["critic_loss"].mean(axis=0), out["metrics"]["entropy"].mean(axis=0), out["metrics"]["ratio"].mean(axis=0)], axis=1)    
    loss_table = wandb.Table(data=loss_table.tolist(), columns=["updates", "total_loss", "actor_loss", "critic_loss", "entropy", "ratio"])'''
    '''print('shape', out["metrics"]["returned_episode_returns"][0].shape)
    updates_x = jnp.arange(out["metrics"]["returned_episode_returns"][0].shape[0])
    returns_table = jnp.stack([updates_x, out["metrics"]["returned_episode_returns"].mean(axis=0)], axis=1)
    returns_table = wandb.Table(data=returns_table.tolist(), columns=["updates", "returns"])
    wandb.log({
        "returns_plot": wandb.plot.line(returns_table, "updates", "returns", title="returns_vs_updates"),
        "returns": out["metrics"]["returned_episode_returns"][:,-1].mean(),
        
    })'''

'''
"total_loss_plot": wandb.plot.line(loss_table, "updates", "total_loss", title="total_loss_vs_updates"),
        "actor_loss_plot": wandb.plot.line(loss_table, "updates", "actor_loss", title="actor_loss_vs_updates"),
        "critic_loss_plot": wandb.plot.line(loss_table, "updates", "critic_loss", title="critic_loss_vs_updates"),
        "entropy_plot": wandb.plot.line(loss_table, "updates", "entropy", title="entropy_vs_updates"),
        "ratio_plot": wandb.plot.line(loss_table, "updates", "ratio", title="ratio_vs_updates"),
'''

if __name__ == "__main__":
    main()
