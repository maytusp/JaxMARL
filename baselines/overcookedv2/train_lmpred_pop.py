'''
Add multi-time-scale predictive representation loss:
hidden_state --> Hidden_Predictor --> pred_hidden_repr
target_hidden_repr = stop_grad(next_hidden_state + gamma_k * next_pred_hidden_repr)
'''
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
import re
import wandb
import functools

import flax.serialization
from flax.core import freeze, unfreeze
from flax import traverse_util

from .utils import OvercookedTransform, OvercookedHeadAlignedTransform


CHECKPOINT_RE = re.compile(r"baseline_seed_(?P<seed>\d+)_step_(?P<step>\d+)\.msgpack$")


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

    def _other_stream_transform(self):
        agent_view_size = self.config["ENV_KWARGS"].get("agent_view_size", 2)
        agent_features_len = self.config.get("AGENT_FEATURES_LEN", 9)
        front_obs = self.config["ENV_KWARGS"].get("front_obs", False)

        if front_obs:
            return OvercookedHeadAlignedTransform(
                agent_view_size=agent_view_size,
                agent_features_len=agent_features_len,
            )
        return OvercookedTransform(
            agent_view_size=agent_view_size,
            agent_features_len=agent_features_len,
        )

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        self_hidden = hidden

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        assert obs.ndim == 5, f"Expected obs (T,B,H,W,C), got {obs.shape}"

        agent_features_len = self.config.get("AGENT_FEATURES_LEN", 9)
        partner_present = jnp.max(
            obs[..., agent_features_len], axis=(-2, -1)
        ) > 0

        if self.config.get("PERSPECTIVE_TRANSFORM", True):
            other_obs = self._other_stream_transform()(obs)
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

        # Reuse only the pretrained CNN feature extractors. The fused recurrent
        # state is trained from scratch on both self and partner-perspective features.
        other_embedding = jnp.where(
            partner_present[..., None],
            other_embedding,
            jnp.zeros_like(other_embedding),
        )

        finetune_self_stream = self.config.get(
            "FINETUNE_SELF_STREAM",
            not self.config.get("STOP_GRAD_SELF", False),
        )
        finetune_other_stream = self.config.get("FINETUNE_OTHER_STREAM", False)

        if not finetune_self_stream:
            self_embedding = jax.lax.stop_gradient(self_embedding)
        if not finetune_other_stream:
            other_embedding = jax.lax.stop_gradient(other_embedding)

        rnn_input = jnp.concatenate([self_embedding, other_embedding], axis=-1)
        self_hidden, embedding = ScannedRNN(name="fusion_rnn")(
            self_hidden, (rnn_input, dones)
        )

        pred_hidden_repr = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="self_pred_fc",
        )(embedding)
        pred_hidden_repr = activation(pred_hidden_repr)
        pred_gammas = tuple(self.config.get("SELF_PRED_GAMMAS", (0.0, 0.5, 0.9)))
        pred_hidden_repr = nn.Dense(
            len(pred_gammas) * embedding.shape[-1],
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="self_pred_out",
        )(pred_hidden_repr)
        pred_hidden_repr = pred_hidden_repr.reshape(
            *embedding.shape[:-1], len(pred_gammas), embedding.shape[-1]
        )

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

        aux = {
            "rnn_hidden": embedding,
            "pred_hidden_repr": pred_hidden_repr,
        }

        return self_hidden, pi, jnp.squeeze(critic, axis=-1), aux
        

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

    if "params" in params:
        params["params"] = target
    return freeze(params)


def _build_trainable_labels(params, config):
    trainable_paths = {
        ("params", "fusion_rnn"),
        ("params", "actor_fc"),
        ("params", "actor_out"),
        ("params", "critic_fc"),
        ("params", "critic_out"),
        ("params", "self_pred_fc"),
        ("params", "self_pred_out"),
    }
    finetune_self_stream = config.get(
        "FINETUNE_SELF_STREAM",
        not config.get("STOP_GRAD_SELF", False),
    )
    finetune_other_stream = config.get("FINETUNE_OTHER_STREAM", False)

    if finetune_self_stream:
        trainable_paths.update(
            {
                ("params", "self_cnn"),
                ("params", "self_ln"),
            }
        )
    if finetune_other_stream:
        trainable_paths.update(
            {
                ("params", "other_cnn"),
                ("params", "other_ln"),
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


def _checkpoint_sort_key(name):
    match = CHECKPOINT_RE.match(os.path.basename(name))
    if match is None:
        return (10**12, 10**12, name)
    return (int(match.group("seed")), int(match.group("step")), name)


def _checkpoint_seed(name):
    match = CHECKPOINT_RE.match(os.path.basename(name))
    if match is None:
        raise ValueError(f"Invalid checkpoint name: {name}")
    return int(match.group("seed"))


def _resolve_layout_checkpoint_dir(config, prefix_key, default_prefix):
    layout = config["ENV_KWARGS"]["layout"]
    prefix = config.get(prefix_key, default_prefix)
    return os.path.join(prefix, layout)


def _discover_checkpoints(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    names = sorted(
        [name for name in os.listdir(checkpoint_dir) if CHECKPOINT_RE.match(name)],
        key=_checkpoint_sort_key,
    )
    if not names:
        raise FileNotFoundError(
            f"No baseline_seed_*_step_*.msgpack checkpoints found in {checkpoint_dir}"
        )
    return names


def _parse_optional_int_sequence(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = value.strip("[]")
        if not value:
            return None
        value = [x.strip() for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _parse_optional_float_sequence(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = value.strip("[]")
        if not value:
            return None
        value = [x.strip() for x in value.split(",") if x.strip()]
    return [float(x) for x in value]


def _select_checkpoint_stages(names, stage_fractions):
    stage_fractions = _parse_optional_float_sequence(stage_fractions)
    if stage_fractions is None:
        return names

    names_by_seed = {}
    for name in names:
        names_by_seed.setdefault(_checkpoint_seed(name), []).append(name)

    selected = []
    for seed in sorted(names_by_seed):
        seed_names = sorted(names_by_seed[seed], key=_checkpoint_sort_key)
        stage_indices = []
        for frac in stage_fractions:
            frac = min(max(frac, 0.0), 1.0)
            idx = int(np.argmin(np.abs(np.linspace(0, 1, len(seed_names)) - frac)))
            stage_indices.append(idx)
        stage_indices = np.unique(np.array(stage_indices, dtype=np.int32))
        selected.extend([seed_names[int(idx)] for idx in stage_indices])
    return sorted(selected, key=_checkpoint_sort_key)


def _limit_checkpoints(names, max_checkpoints):
    if max_checkpoints is None:
        return names

    max_checkpoints = int(max_checkpoints)
    if max_checkpoints <= 0 or max_checkpoints >= len(names):
        return names

    indices = np.linspace(0, len(names) - 1, max_checkpoints)
    indices = np.round(indices).astype(np.int32)
    indices = np.unique(indices)
    if len(indices) < max_checkpoints:
        remaining = [idx for idx in range(len(names)) if idx not in set(indices)]
        indices = np.array(
            list(indices) + remaining[: max_checkpoints - len(indices)],
            dtype=np.int32,
        )
    return [names[int(idx)] for idx in sorted(indices[:max_checkpoints])]


def _filter_checkpoints_by_seed(names, seed_subset):
    seed_subset = _parse_optional_int_sequence(seed_subset)
    if seed_subset is None:
        return names

    seed_subset = set(seed_subset)
    selected = [name for name in names if _checkpoint_seed(name) in seed_subset]
    if not selected:
        raise ValueError(
            f"POP_PARTNER_SEEDS={sorted(seed_subset)} did not match any partner checkpoints"
        )
    return selected


def _seed_final_checkpoint_names(names, num_seeds):
    final_names = []
    for seed_id in range(num_seeds):
        seed_prefix = f"baseline_seed_{seed_id}_"
        seed_names = [
            name for name in names if os.path.basename(name).startswith(seed_prefix)
        ]
        if not seed_names:
            raise FileNotFoundError(f"No init checkpoint found for seed {seed_id}")
        final_names.append(sorted(seed_names, key=_checkpoint_sort_key)[-1])
    return final_names


def make_dummy_twostream_params(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)
    network = TwoStreamActorCriticRNN(
        env.action_space(env.agents[0]).n, config=config
    )

    rng = jax.random.PRNGKey(config.get("SEED", 0))
    rng, reset_rng, init_rng = jax.random.split(rng, 3)
    reset_rng = jax.random.split(reset_rng, config["NUM_ENVS"])
    obsv_init, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    init_hstate = ScannedRNN.initialize_carry(
        config["NUM_ENVS"], 2 * config["GRU_HIDDEN_DIM"]
    )
    init_x = (
        obsv_init[env.agents[0]][jnp.newaxis, ...],
        jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
    )
    return network.init(init_rng, init_hstate, init_x)


def load_init_pool(config, dummy_params):
    init_dir = _resolve_layout_checkpoint_dir(
        config, "INIT_CHECKPOINTS_PREFIX", "checkpoints/ph2v4/"
    )
    names = config.get("INIT_CHECKPOINTS")
    if names is None:
        names = _seed_final_checkpoint_names(
            _discover_checkpoints(init_dir),
            config["NUM_SEEDS"],
        )
    else:
        names = list(names)

    loaded_params = []
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(init_dir, name)
        with open(path, "rb") as f:
            loaded_params.append(flax.serialization.from_bytes(dummy_params, f.read()))

    stacked_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *loaded_params
    )
    print(
        f"Loaded {len(names)} LMpred init checkpoints from {init_dir}: "
        f"{', '.join(os.path.basename(name) for name in names)}"
    )
    return {"params": stacked_params, "names": names}


def load_partner_pool(config, dummy_params, init_names):
    partner_dir = _resolve_layout_checkpoint_dir(
        config, "PARTNER_CHECKPOINTS_PREFIX", "checkpoints/ph2v4/"
    )
    names = config.get("POP_PARTNER_CHECKPOINTS")
    if names is None:
        names = _discover_checkpoints(partner_dir)
        names = _filter_checkpoints_by_seed(names, config.get("POP_PARTNER_SEEDS"))
        names = _select_checkpoint_stages(
            names,
            config.get("POP_PARTNER_STAGE_FRACTIONS", [1.0]),
        )
        names = _limit_checkpoints(names, config.get("POP_MAX_PARTNERS"))
    names = list(names)

    loaded_params = []
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(partner_dir, name)
        with open(path, "rb") as f:
            loaded_params.append(flax.serialization.from_bytes(dummy_params, f.read()))

    exclude_self = bool(config.get("POP_EXCLUDE_INIT_SELF", True))
    partner_basenames = [os.path.basename(name) for name in names]
    masks = []
    for seed_id, init_name in enumerate(init_names):
        if exclude_self:
            allowed = [
                os.path.basename(partner_name) != os.path.basename(init_name)
                for partner_name in names
            ]
        else:
            allowed = [True] * len(names)
        if not any(allowed):
            raise ValueError(
                "POP_EXCLUDE_INIT_SELF removed every partner for "
                f"seed {seed_id}; add more partner checkpoints or disable exclusion."
            )
        masks.append(allowed)

    stacked_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *loaded_params
    )
    seeds = sorted({_checkpoint_seed(name) for name in names})
    steps = sorted({_checkpoint_step(name) for name in names})
    print(
        f"Loaded {len(names)} frozen LMpred partners from {partner_dir} "
        f"({len(seeds)} seeds, {len(steps)} steps; max step {max(steps)})"
    )
    print(f"LMpred partner pool: {', '.join(partner_basenames)}")
    return {
        "params": stacked_params,
        "names": names,
        "allowed_masks": jnp.asarray(masks, dtype=bool),
        "seeds": seeds,
        "steps": steps,
    }


def make_train(config, init_pool, partner_pool):
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

        # INIT NETWORK
        ego_network = TwoStreamActorCriticRNN(
            env.action_space(env.agents[0]).n, config=config
        )
        partner_network = TwoStreamActorCriticRNN(
            env.action_space(env.agents[1]).n, config=config
        )
        partner_pool_params = partner_pool["params"]
        partner_allowed_mask = partner_pool["allowed_masks"][seed_idx]
        num_partners = jax.tree_util.tree_leaves(partner_pool_params)[0].shape[0]

        rng, _rng_reset, _rng_init = jax.random.split(rng, 3)

        reset_rng = jax.random.split(_rng_reset, config["NUM_ENVS"])
        obsv_init, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        obs0_init = obsv_init[env.agents[0]]
        init_x = (
            obs0_init[jnp.newaxis, ...],  # (1, NUM_ENVS, H, W, C)
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
        )

        fusion_hidden_dim = 2 * config["GRU_HIDDEN_DIM"]
        ego_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], fusion_hidden_dim
        )
        network_params = ego_network.init(_rng_init, ego_init_hstate, init_x)
        network_params = freeze(
            jax.tree_util.tree_map(lambda x: x[seed_idx], init_pool["params"])
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
        ego_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], fusion_hidden_dim
        )
        partner_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], fusion_hidden_dim
        )

        pop_priority_alpha = float(config.get("POP_PRIORITY_ALPHA", 3.0))
        pop_priority_eps = float(config.get("POP_PRIORITY_EPS", 1e-6))
        pop_score_update_rate = float(config.get("POP_SCORE_UPDATE_RATE", 0.1))
        pop_prioritized_sampling = bool(config.get("POP_PRIORITIZED_SAMPLING", True))

        def _rank_priorities(scores, allowed_mask):
            order = jnp.argsort(scores)
            ranks = jnp.zeros_like(scores)
            ranks = ranks.at[order].set(
                jnp.arange(scores.shape[0], 0, -1, dtype=scores.dtype)
            )
            priorities = jnp.power(ranks + pop_priority_eps, pop_priority_alpha)
            priorities = priorities * allowed_mask.astype(priorities.dtype)
            return priorities / jnp.maximum(jnp.sum(priorities), pop_priority_eps)

        def _sample_partner_indices(rng, scores):
            if pop_prioritized_sampling:
                return jax.random.choice(
                    rng,
                    num_partners,
                    shape=(config["NUM_ENVS"],),
                    p=_rank_priorities(scores, partner_allowed_mask),
                )
            allowed = partner_allowed_mask.astype(jnp.float32)
            probs = allowed / jnp.maximum(jnp.sum(allowed), 1.0)
            return jax.random.choice(
                rng,
                num_partners,
                shape=(config["NUM_ENVS"],),
                p=probs,
            )

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
                    partner_scores,
                    episode_return,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng_ego, _rng_partner, _rng_new_partner_id = jax.random.split(
                    rng, 4
                )

                ego_obs = last_obs[env.agents[0]]
                ac_in = (
                    ego_obs[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                ego_hstate, pi, value, _ = ego_network.apply(
                    train_state.params, ego_hstate, ac_in
                )
                ego_action = pi.sample(seed=_rng_ego).squeeze(0)
                log_prob = pi.log_prob(ego_action)

                selected_partner_params = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, partner_idx, axis=0), partner_pool_params
                )

                def _partner_act(params, hstate, obs, done, rng):
                    partner_in = (
                        obs[jnp.newaxis, jnp.newaxis, ...],
                        done[jnp.newaxis, jnp.newaxis],
                    )
                    next_hstate, partner_pi, _, _ = partner_network.apply(
                        params, hstate[jnp.newaxis, :], partner_in
                    )
                    partner_action = partner_pi.sample(seed=rng).squeeze()
                    return next_hstate.squeeze(0), partner_action

                partner_rng = jax.random.split(_rng_partner, config["NUM_ENVS"])
                partner_hstate, partner_action = jax.vmap(_partner_act)(
                    selected_partner_params,
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
                original_reward = jnp.array([reward[a] for a in env.agents])

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward

                info = jax.tree_util.tree_map(
                    lambda x: x[0]
                    if x.ndim > 0 and x.shape[0] == env.num_agents
                    else x,
                    info,
                )
                info["shaped_reward"] = shaped_reward[0]
                info["original_reward"] = original_reward[0]
                info["anneal_factor"] = jnp.full(
                    (config["NUM_ENVS"],), anneal_factor
                )
                info["combined_reward"] = combined_reward[0]
                priority_probs = _rank_priorities(partner_scores, partner_allowed_mask)
                info["pop_partner_score_mean"] = jnp.full(
                    (config["NUM_ENVS"],), jnp.mean(partner_scores)
                )
                info["pop_priority_entropy"] = jnp.full(
                    (config["NUM_ENVS"],),
                    -jnp.sum(
                        priority_probs
                        * jnp.log(jnp.maximum(priority_probs, pop_priority_eps))
                    ),
                )
                done_batch = done["__all__"]
                completed_return = episode_return + reward[env.agents[0]]
                done_float = done_batch.astype(jnp.float32)
                partner_counts = jnp.bincount(
                    partner_idx,
                    weights=done_float,
                    length=num_partners,
                )
                partner_return_sums = jnp.bincount(
                    partner_idx,
                    weights=done_float * completed_return,
                    length=num_partners,
                )
                partner_return_means = partner_return_sums / jnp.maximum(
                    partner_counts,
                    1.0,
                )
                partner_scores = jnp.where(
                    partner_counts > 0,
                    (1.0 - pop_score_update_rate) * partner_scores
                    + pop_score_update_rate * partner_return_means,
                    partner_scores,
                )
                episode_return = jnp.where(done_batch, 0.0, completed_return)
                new_partner_idx = _sample_partner_indices(
                    _rng_new_partner_id,
                    partner_scores,
                )
                partner_idx = jnp.where(done_batch, new_partner_idx, partner_idx)
                transition = Transition(
                    done_batch,
                    ego_action,
                    value.squeeze(0),
                    reward[env.agents[0]],
                    log_prob.squeeze(0),
                    ego_obs,
                    info,
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
                    partner_scores,
                    episode_return,
                    rng,
                )
                return runner_state, transition

            initial_hstate = runner_state[5]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                ego_hstate,
                partner_hstate,
                partner_idx,
                partner_scores,
                episode_return,
                rng,
            ) = runner_state
            last_obs_batch = last_obs[env.agents[0]]
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val, _ = ego_network.apply(train_state.params, ego_hstate, ac_in)
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
                        _, pi, value, aux = ego_network.apply(
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
                        approx_kl = (traj_batch.log_prob - log_prob).mean()
                        clip_frac = (
                            jnp.abs(ratio - 1.0) > config["CLIP_EPS"]
                        ).astype(jnp.float32).mean()
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

                        hidden = aux["rnn_hidden"]
                        pred_hidden_repr = aux["pred_hidden_repr"]
                        next_hidden = jnp.concatenate(
                            [hidden[1:], jnp.zeros_like(hidden[-1:])], axis=0
                        )
                        next_pred_hidden_repr = jnp.concatenate(
                            [
                                pred_hidden_repr[1:],
                                jnp.zeros_like(pred_hidden_repr[-1:]),
                            ],
                            axis=0,
                        )
                        not_done = 1.0 - traj_batch.done.astype(jnp.float32)
                        has_next_step = jnp.ones_like(not_done).at[-1].set(0.0)
                        future_mask = (not_done * has_next_step)[..., None, None]
                        pred_gammas = jnp.asarray(
                            config.get("SELF_PRED_GAMMAS", (0.0, 0.5, 0.9)),
                            dtype=hidden.dtype,
                        ).reshape((1, 1, -1, 1))
                        pred_target = jax.lax.stop_gradient(
                            future_mask
                            * (
                                next_hidden[..., None, :]
                                + pred_gammas * next_pred_hidden_repr
                            )
                        )
                        pred_error_clip = config.get("SELF_PRED_ERROR_CLIP", 10.0)
                        pred_error = jnp.clip(
                            pred_hidden_repr - pred_target,
                            -pred_error_clip,
                            pred_error_clip,
                        )
                        pred_delta = config.get("SELF_PRED_HUBER_DELTA", 1.0)
                        pred_loss = optax.huber_loss(
                            pred_error, jnp.zeros_like(pred_error), delta=pred_delta
                        ).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config.get("SELF_PRED_COEF", 0.1) * pred_loss
                        )
                        loss_metrics = {
                            "loss_total": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "self_pred_loss": pred_loss,
                            "approx_kl": approx_kl,
                            "clip_frac": clip_frac,
                            "ratio_mean": ratio.mean(),
                            "ratio_max": ratio.max(),
                        }
                        return total_loss, loss_metrics

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (_, loss_metrics), grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_metrics

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
            for key, value in loss_info.items():
                metric[f"ppo/{key}"] = value.mean()
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            # --- LMpred population checkpoint saving ---
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
                checkpoints_prefix = config.get(
                    "CHECKPOINTS_PREFIX", "checkpoints/lmpred_pop/"
                )
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
                partner_scores,
                episode_return,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        _rng, _rng_partner_idx = jax.random.split(_rng)
        partner_scores = jnp.ones((num_partners,), dtype=jnp.float32)
        episode_return = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32)
        partner_idx = _sample_partner_indices(_rng_partner_idx, partner_scores)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            ego_init_hstate,
            partner_init_hstate,
            partner_idx,
            partner_scores,
            episode_return,
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
    config.setdefault("INIT_CHECKPOINTS_PREFIX", "checkpoints/ph2v4/")
    config.setdefault("PARTNER_CHECKPOINTS_PREFIX", "checkpoints/ph2v4/")
    config.setdefault("CHECKPOINTS_PREFIX", "checkpoints/lmpred_pop/")
    config.setdefault("POP_PARTNER_STAGE_FRACTIONS", [1.0])
    config.setdefault("POP_EXCLUDE_INIT_SELF", True)
    config.setdefault("POP_PRIORITIZED_SAMPLING", True)
    config.setdefault("POP_PRIORITY_ALPHA", 3.0)
    config.setdefault("POP_PRIORITY_EPS", 1e-6)
    config.setdefault("POP_SCORE_UPDATE_RATE", 0.1)

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    checkpoints_prefix = config.get("CHECKPOINTS_PREFIX", "")
    checkpoint_model_name = os.path.basename(os.path.normpath(checkpoints_prefix))
    model_name = config.get("MODEL_NAME", checkpoint_model_name or "lmpred_pop")
    if config["ENV_KWARGS"].get("front_obs", True):
        model_name += "_obsfront"
    perspective_transform = config.get("PERSPECTIVE_TRANSFORM", True)
    if perspective_transform:
        model_name += "_cpt"
    elif not(perspective_transform):
        model_name += "_sameinp"
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["LMpredPop", "BR", "IPPO", "RNN", "OvercookedV2"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"{model_name}_{layout_name}",
    )

    with jax.disable_jit(False):
        dummy_params = make_dummy_twostream_params(config)
        init_pool = load_init_pool(config, dummy_params)
        partner_pool = load_partner_pool(config, dummy_params, init_pool["names"])
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config, init_pool, partner_pool))
        seed_ids = jnp.arange(num_seeds)
        out = jax.vmap(train_jit)(rngs, seed_ids)


if __name__ == "__main__":
    main()
