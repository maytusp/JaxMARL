# Prepare diverse partners (Phase 1 in FCP\\\)
import jax
import jax.numpy as jnp
import flax.linen as nn
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

import flax.serialization
from flax.core import freeze, unfreeze
from flax import traverse_util

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
        assert x.ndim == 4, f"CNN expected (B,H,W,C), got {x.shape}"
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

        assert obs.ndim == 5, f"Expected obs (T,B,H,W,C), got {obs.shape}"

        h, w, c = obs.shape[-3:]
        flat_obs = obs.reshape(-1, h, w, c)

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )

        embedding = embed_model(flat_obs)
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
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class TwoStreamActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        self_hidden, other_hidden = hidden

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        assert obs.ndim == 5, f"Expected obs (T,B,H,W,C), got {obs.shape}"

        if self.config.get("PERSPECTIVE_TRANSFORM", True):
            other_obs = OvercookedToMTransform(
                agent_view_size=self.config["ENV_KWARGS"].get("agent_view_size", 2),
                agent_features_len=self.config.get("AGENT_FEATURES_LEN", 9),
            )(obs)
        else:
            other_obs = obs

        h, w, c = obs.shape[-3:]
        flat_obs = obs.reshape(-1, h, w, c)
        flat_other_obs = other_obs.reshape(-1, h, w, c)

        self_embedding = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="self_cnn",
        )(flat_obs)
        other_embedding = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="other_cnn",
        )(flat_other_obs)

        self_embedding = self_embedding.reshape(*obs.shape[:-3], -1)
        other_embedding = other_embedding.reshape(*obs.shape[:-3], -1)

        self_embedding = nn.LayerNorm(name="self_ln")(self_embedding)
        other_embedding = nn.LayerNorm(name="other_ln")(other_embedding)

        self_hidden, self_embedding = ScannedRNN(name="self_rnn")(
            self_hidden, (self_embedding, dones)
        )
        other_hidden, other_embedding = ScannedRNN(name="other_rnn")(
            other_hidden, (other_embedding, dones)
        )

        finetune_self_stream = self.config.get(
            "FINETUNE_SELF_STREAM",
            not self.config.get("STOP_GRAD_SELF", False),
        )
        finetune_other_stream = self.config.get(
            "FINETUNE_OTHER_STREAM",
            not self.config.get("STOP_GRAD_OTHER", False),
        )

        if not finetune_self_stream:
            self_embedding = jax.lax.stop_gradient(self_embedding)
        if not finetune_other_stream:
            other_embedding = jax.lax.stop_gradient(other_embedding)

        embedding = jnp.concatenate([self_embedding, other_embedding], axis=-1)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor_fc",
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic_fc",
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(critic)

        return (self_hidden, other_hidden), pi, jnp.squeeze(critic, axis=-1)
        

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


def _params_subtree(variables):
    if isinstance(variables, dict) or hasattr(variables, "keys"):
        return variables["params"] if "params" in variables else variables
    return variables


def _copy_single_trunk_to_two_stream(two_stream_params, single_params):
    params = unfreeze(two_stream_params)
    single = unfreeze(_params_subtree(single_params))
    target = params["params"] if "params" in params else params

    target["self_cnn"] = single["CNN_0"]
    target["other_cnn"] = single["CNN_0"]
    target["self_ln"] = single["LayerNorm_0"]
    target["other_ln"] = single["LayerNorm_0"]
    target["self_rnn"] = single["ScannedRNN_0"]
    target["other_rnn"] = single["ScannedRNN_0"]

    if "params" in params:
        params["params"] = target
    return freeze(params)


def _build_trainable_labels(params, config):
    trainable_paths = {
        ("params", "actor_fc"),
        ("params", "actor_out"),
        ("params", "critic_fc"),
        ("params", "critic_out"),
    }
    finetune_self_stream = config.get(
        "FINETUNE_SELF_STREAM",
        not config.get("STOP_GRAD_SELF", False),
    )
    finetune_other_stream = config.get(
        "FINETUNE_OTHER_STREAM",
        not config.get("STOP_GRAD_OTHER", False),
    )

    if finetune_self_stream:
        trainable_paths.update(
            {
                ("params", "self_cnn"),
                ("params", "self_ln"),
                ("params", "self_rnn"),
            }
        )
    if finetune_other_stream:
        trainable_paths.update(
            {
                ("params", "other_cnn"),
                ("params", "other_ln"),
                ("params", "other_rnn"),
            }
        )

    flat_params = traverse_util.flatten_dict(unfreeze(params))
    flat_labels = {}
    for key in flat_params:
        flat_labels[key] = (
            "train"
            if any(key[: len(path)] == path for path in trainable_paths)
            else "freeze"
        )
    return freeze(traverse_util.unflatten_dict(flat_labels))


def _checkpoint_step(ckpt_name):
    stem = os.path.basename(ckpt_name).replace(".msgpack", "")
    if "_step_" not in stem:
        return -1
    return int(stem.rsplit("_step_", 1)[-1])


def _resolve_checkpoint_dir(config):
    layout = config["ENV_KWARGS"]["layout"]
    prefix = config.get(
        "OLD_SELF_CHECKPOINTS_PREFIX",
        config.get("PRETRAINED_CHECKPOINTS_PREFIX", "./checkpoints/single"),
    )
    return os.path.join(prefix, layout)


def _seed_checkpoint_names(config, seed_id, pool_dir):
    by_seed = config.get("FCP_CHECKPOINTS_BY_SEED")
    if by_seed is not None:
        names = by_seed.get(str(seed_id), by_seed.get(seed_id))
        if names is None:
            raise ValueError(f"No FCP_CHECKPOINTS_BY_SEED entry for seed {seed_id}")
        return list(names)

    names = config.get("OLD_SELF_CHECKPOINTS", config.get("FCP_CHECKPOINTS"))
    if names is not None:
        names = list(names)
        seed_prefix = f"baseline_seed_{seed_id}_"
        seed_names = [n for n in names if os.path.basename(n).startswith(seed_prefix)]
        return seed_names if seed_names else names

    seed_prefix = f"baseline_seed_{seed_id}_"
    discovered = [
        name
        for name in os.listdir(pool_dir)
        if name.startswith(seed_prefix) and name.endswith(".msgpack")
    ]
    return sorted(discovered, key=_checkpoint_step)


def load_old_self_pool(config, dummy_single_params):
    pool_dir = _resolve_checkpoint_dir(config)
    num_seeds = config["NUM_SEEDS"]
    max_partners = config.get("MAX_OLD_SELF_PARTNERS")

    per_seed_params = []
    per_seed_names = []
    expected_pool_size = None
    for seed_id in range(num_seeds):
        names = _seed_checkpoint_names(config, seed_id, pool_dir)
        if max_partners is not None:
            names = names[-int(max_partners) :]
        if not names:
            raise FileNotFoundError(
                f"No old self checkpoints found for seed {seed_id} in {pool_dir}"
            )
        if expected_pool_size is None:
            expected_pool_size = len(names)
        elif len(names) != expected_pool_size:
            raise ValueError(
                "Each seed must load the same number of old self checkpoints for vmap. "
                f"Seed 0 has {expected_pool_size}, seed {seed_id} has {len(names)}."
            )

        loaded = []
        for name in names:
            path = name if os.path.isabs(name) else os.path.join(pool_dir, name)
            with open(path, "rb") as f:
                loaded.append(flax.serialization.from_bytes(dummy_single_params, f.read()))
        per_seed_params.append(
            jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *loaded)
        )
        per_seed_names.append(names)
        print(f"Loaded {len(names)} old self checkpoints for seed {seed_id} from {pool_dir}")

    stacked_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *per_seed_params
    )
    return {"params": stacked_params, "names": per_seed_names}


def make_train(config, old_self_pool):
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

    def train(rng, seed_idx):

        # INIT NETWORKS
        ego_network = TwoStreamActorCriticRNN(
            env.action_space(env.agents[0]).n, config=config
        )
        old_self_network = ActorCriticRNN(
            env.action_space(env.agents[0]).n, config=config
        )
        old_self_params = jax.tree_util.tree_map(lambda x: x[seed_idx], old_self_pool["params"])
        num_old_self_partners = jax.tree_util.tree_leaves(old_self_params)[0].shape[0]

        def _get_old_self_params(params_tree, partner_idx):
            return jax.tree_util.tree_map(lambda x: x[partner_idx], params_tree)

        rng, _rng_reset, _rng_init = jax.random.split(rng, 3)

        reset_rng = jax.random.split(_rng_reset, config["NUM_ENVS"])
        obsv_init, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        obs0_init = obsv_init[env.agents[0]]
        init_x = (
            obs0_init[jnp.newaxis, ...],  # (1, NUM_ENVS, H, W, C)
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
        )

        stream_init_idx = config.get("STREAM_INIT_PARTNER_INDEX", -1)
        trunk_source_params = _get_old_self_params(old_self_params, stream_init_idx)

        ego_init_hstate = (
            ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
            ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
        )
        network_params = ego_network.init(_rng_init, ego_init_hstate, init_x)
        network_params = _copy_single_trunk_to_two_stream(
            network_params, trunk_source_params
        )
        
        if config["ANNEAL_LR"]:
            base_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            base_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        tx = optax.multi_transform(
            {
                "train": base_tx,
                "freeze": optax.set_to_zero(),
            },
            _build_trainable_labels(network_params, config),
        )
        train_state = TrainState.create(
            apply_fn=ego_network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ego_init_hstate = (
            ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
            ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]),
        )
        partner_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        rng, _rng_partner = jax.random.split(rng)
        partner_idx = jax.random.randint(
            _rng_partner,
            (config["NUM_ENVS"],),
            minval=0,
            maxval=num_old_self_partners,
        )

        def _get_old_self_params_batch(params_tree, partner_idx):
            return jax.tree_util.tree_map(lambda x: x[partner_idx], params_tree)

        def _apply_old_self_partner(params, hidden, obs, done, rng):
            next_hidden, pi, _ = old_self_network.apply(
                params,
                hidden[jnp.newaxis, :],
                (obs[jnp.newaxis, jnp.newaxis, ...], done[jnp.newaxis, jnp.newaxis]),
            )
            action = pi.sample(seed=rng).squeeze()
            return next_hidden.squeeze(0), action

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
                    ego_hstate,
                    partner_hstate,
                    partner_idx,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                ego_obs_batch = last_obs[env.agents[0]]
                ac_in = (
                    ego_obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                ego_hstate, pi, value = ego_network.apply(
                    train_state.params, ego_hstate, ac_in
                )
                ego_action = pi.sample(seed=_rng).squeeze(0)
                log_prob = pi.log_prob(ego_action).squeeze(0)
                value = value.squeeze(0)

                rng, _rng_partner = jax.random.split(rng)
                partner_rng = jax.random.split(_rng_partner, config["NUM_ENVS"])
                partner_params_batch = _get_old_self_params_batch(
                    old_self_params, partner_idx
                )
                partner_hstate, partner_action = jax.vmap(_apply_old_self_partner)(
                    partner_params_batch,
                    partner_hstate,
                    last_obs[env.agents[1]],
                    last_done,
                    partner_rng,
                )

                env_act = {
                    env.agents[0]: ego_action,
                    env.agents[1]: partner_action,
                }

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                original_reward = reward[env.agents[0]]

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                shaped_reward = info["shaped_reward"][env.agents[0]]
                combined_reward = reward[env.agents[0]]

                metric_info = {k: v for k, v in info.items() if k != "shaped_reward"}
                metric_info["shaped_reward"] = shaped_reward
                metric_info["original_reward"] = original_reward
                metric_info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                metric_info["combined_reward"] = combined_reward
                metric_info["partner_idx"] = partner_idx

                def _ego_metric(x):
                    if x.ndim > 1 and x.shape[-1] == env.num_agents:
                        x = x[..., 0]
                    return x.reshape((config["NUM_ENVS"],))

                metric_info = jax.tree_util.tree_map(_ego_metric, metric_info)
                done_batch = done["__all__"]
                rng, _rng_partner_reset = jax.random.split(rng)
                new_partner_idx = jax.random.randint(
                    _rng_partner_reset,
                    (config["NUM_ENVS"],),
                    minval=0,
                    maxval=num_old_self_partners,
                )
                partner_idx = jnp.where(done_batch, new_partner_idx, partner_idx)
                transition = Transition(
                    done_batch,
                    ego_action,
                    value,
                    combined_reward,
                    log_prob,
                    ego_obs_batch,
                    metric_info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    ego_hstate,
                    partner_hstate,
                    partner_idx,
                    rng,
                )
                return runner_state, transition

            initial_hstate = runner_state[-4]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, update_step, ego_hstate, partner_hstate, partner_idx, rng = (
                runner_state
            )
            last_obs_batch = last_obs[env.agents[0]]
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = ego_network.apply(train_state.params, ego_hstate, ac_in)
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
                        _, pi, value = ego_network.apply(
                            params,
                            jax.tree_util.tree_map(lambda h: h.squeeze(), init_hstate),
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
                    lambda h: jnp.reshape(h, (1, config["NUM_ACTORS"], -1)),
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
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            # --- FCP SKILL-DIVERSE CHECKPOINT SAVING ---
            def save_checkpoint(step_scalar, params, seed_scalar):
                current_step = int(step_scalar)
                seed_id = int(seed_scalar)

                num_skill_levels = config.get("NUM_SKILL_LEVELS", 10)
                save_interval = max(1, config["NUM_UPDATES"] // num_skill_levels)

                is_first_step = (current_step <= 1)
                is_interval_step = (current_step % save_interval == 0)

                if not (is_first_step or is_interval_step):
                    return

                layout = config["ENV_KWARGS"]["layout"]
                checkpoints_prefix = config.get("CHECKPOINTS_PREFIX", "./fcp_pool")
                save_dir = os.path.join(checkpoints_prefix, layout)
                os.makedirs(save_dir, exist_ok=True)

                single_seed_params = jax.tree_util.tree_map(lambda x: np.array(x), params)
                bytes_data = flax.serialization.to_bytes(single_seed_params)

                file_path = os.path.join(
                    save_dir,
                    f"baseline_seed_{seed_id}_step_{current_step}.msgpack"
                )
                with open(file_path, "wb") as f:
                    f.write(bytes_data)

                print(f"--> Saved seed {seed_id} checkpoint at step {current_step} to {file_path}")

            # Execute the callback unconditionally
            jax.debug.callback(save_checkpoint, update_step, train_state.params, seed_idx)


            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                ego_hstate,
                partner_hstate,
                partner_idx,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            ego_init_hstate,
            partner_init_hstate,
            partner_idx,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(
    version_base=None, config_path="", config_name=""
)
def main(config):
    config = OmegaConf.to_container(config)

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    model_name = "ppo"
    perspective_transform = config.get("PERSPECTIVE_TRANSFORM", True)
    if perspective_transform:
        model_name += "_cpt"
    elif not(perspective_transform):
        model_name += "_sameinp"
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "OvercookedV2"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"{model_name}_{layout_name}",
    )

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        dummy_env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        dummy_network = ActorCriticRNN(
            dummy_env.action_space(dummy_env.agents[0]).n, config=config
        )
        dummy_reset_rng = jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"])
        dummy_obsv, _ = jax.vmap(dummy_env.reset, in_axes=(0,))(dummy_reset_rng)
        dummy_obs = dummy_obsv[dummy_env.agents[0]]
        dummy_x = (
            dummy_obs[jnp.newaxis, ...],
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
        )
        dummy_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        dummy_params = dummy_network.init(
            jax.random.PRNGKey(0), dummy_hstate, dummy_x
        )
        old_self_pool = load_old_self_pool(config, dummy_params)
        train_jit = jax.jit(make_train(config, old_self_pool))
        seed_ids = jnp.arange(num_seeds)
        out = jax.vmap(train_jit)(rngs, seed_ids)


if __name__ == "__main__":
    main()
