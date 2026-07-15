"""LMPred training for JaxRobotarium level-based foraging.

This is a minimal flat-observation adaptation of
``baselines.overcookedv2.train_lmpred``. The PPO/RNN/self-prediction training
logic is intentionally kept close to the Overcooked version, while the CNN and
grid perspective transform are replaced with Foraging-specific vector encoders.
"""

import functools
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Sequence, Tuple, Union

import distrax
import flax.serialization
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import struct, traverse_util
from flax.core import freeze, unfreeze
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

# Make the vendored JaxRobotarium importable when running from the JaxMARL root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_JAXROBOTARIUM = _REPO_ROOT / "jaxmarl" / "environments" / "JaxRobotarium"
if _LOCAL_JAXROBOTARIUM.exists():
    sys.path.insert(0, str(_LOCAL_JAXROBOTARIUM))

import jaxmarl


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
        rnn_state = carry
        ins, resets = x

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])
        rnn_state = jnp.where(resets[:, None], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ForagingPerspectiveTransform(nn.Module):
    """Build partner-perspective Foraging observations by swapping vector slots.

    Expected 3-agent, 2-resource, full-capability-set layout:
    [ego pose, partner1 pose, partner2 pose, resource info, ego cap, partner caps].
    """

    num_agents: int = 3
    num_resources: int = 2
    pose_dim: int = 3
    resource_dim: int = 3
    capability_dim: int = 1

    def _swap_with_partner(self, obs: jnp.ndarray, partner_idx: int) -> jnp.ndarray:
        leading_shape = obs.shape[:-1]
        pose_width = self.num_agents * self.pose_dim
        resource_width = self.num_resources * self.resource_dim
        capability_width = self.num_agents * self.capability_dim
        expected_width = pose_width + resource_width + capability_width
        assert (
            obs.shape[-1] == expected_width
        ), f"Expected Foraging obs dim {expected_width}, got {obs.shape[-1]}"

        pose_end = pose_width
        resource_end = pose_end + resource_width

        poses = obs[..., :pose_end].reshape(
            *leading_shape, self.num_agents, self.pose_dim
        )
        resources = obs[..., pose_end:resource_end]
        capabilities = obs[..., resource_end:].reshape(
            *leading_shape, self.num_agents, self.capability_dim
        )

        swapped_poses = poses.at[..., 0, :].set(poses[..., partner_idx, :])
        swapped_poses = swapped_poses.at[..., partner_idx, :].set(poses[..., 0, :])

        swapped_capabilities = capabilities.at[..., 0, :].set(
            capabilities[..., partner_idx, :]
        )
        swapped_capabilities = swapped_capabilities.at[..., partner_idx, :].set(
            capabilities[..., 0, :]
        )

        return jnp.concatenate(
            [
                swapped_poses.reshape(*leading_shape, pose_width),
                resources,
                swapped_capabilities.reshape(*leading_shape, capability_width),
            ],
            axis=-1,
        )

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        assert self.num_agents == 3, "Foraging LMPred perspective transform is 3-agent only."
        partner1_obs = self._swap_with_partner(obs, 1)
        partner2_obs = self._swap_with_partner(obs, 2)
        return jnp.stack([partner1_obs, partner2_obs], axis=-2)


class MLPEncoder(nn.Module):
    output_size: int
    hidden_size: int
    num_layers: int
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = self.activation(x)

        x = nn.Dense(
            self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        return self.activation(x)


class TwoStreamForagingActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        assert obs.ndim == 3, f"Expected obs (T,B,D), got {obs.shape}"

        num_agents = self.config["ENV_KWARGS"]["num_agents"]
        num_resources = self.config["ENV_KWARGS"]["num_resources"]
        capability_dim = self.config.get("CAPABILITY_DIM", 1)
        assert num_agents == 3, "This Foraging LMPred script supports exactly 3 agents."

        transform = ForagingPerspectiveTransform(
            num_agents=num_agents,
            num_resources=num_resources,
            capability_dim=capability_dim,
        )
        if self.config.get("PERSPECTIVE_TRANSFORM", True):
            partner_obs = transform(obs)
        else:
            partner_obs = jnp.stack([obs, obs], axis=-2)

        encoder_hidden = self.config.get("ENCODER_HIDDEN_DIM", self.config["FC_DIM_SIZE"])
        encoder_layers = self.config.get("ENCODER_NUM_LAYERS", 2)

        flat_obs = obs.reshape(-1, obs.shape[-1])
        self_embedding = MLPEncoder(
            output_size=self.config["GRU_HIDDEN_DIM"],
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            activation=activation,
            name="self_mlp",
        )(flat_obs)
        self_embedding = self_embedding.reshape(*obs.shape[:-1], -1)
        self_embedding = nn.LayerNorm(name="self_ln")(self_embedding)

        flat_partner_obs = partner_obs.reshape(-1, obs.shape[-1])
        partner_embedding = MLPEncoder(
            output_size=self.config["GRU_HIDDEN_DIM"],
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            activation=activation,
            name="partner_mlp",
        )(flat_partner_obs)
        partner_embedding = partner_embedding.reshape(
            *obs.shape[:-1], 2, self.config["GRU_HIDDEN_DIM"]
        )
        partner_embedding = nn.LayerNorm(name="partner_ln")(partner_embedding)

        finetune_self_stream = self.config.get(
            "FINETUNE_SELF_STREAM",
            not self.config.get("STOP_GRAD_SELF", False),
        )
        finetune_partner_stream = self.config.get("FINETUNE_OTHER_STREAM", False)

        if not finetune_self_stream:
            self_embedding = jax.lax.stop_gradient(self_embedding)
        if not finetune_partner_stream:
            partner_embedding = jax.lax.stop_gradient(partner_embedding)

        partner_embedding = partner_embedding.reshape(*obs.shape[:-1], -1)
        rnn_input = jnp.concatenate([self_embedding, partner_embedding], axis=-1)
        hidden, embedding = ScannedRNN(name="fusion_rnn")(hidden, (rnn_input, dones))

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
        return hidden, pi, jnp.squeeze(critic, axis=-1), aux


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@struct.dataclass
class RobotariumLogEnvState:
    env_state: Any
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray


class RobotariumLogWrapper:
    """Episode logging wrapper for JaxRobotarium envs with reset(key)."""

    def __init__(self, env, replace_info: bool = False):
        self._env = env
        self.replace_info = replace_info

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x: Dict[str, jnp.ndarray]):
        return jnp.stack([x[a] for a in self._env.agents])

    def reset(self, key: jax.Array) -> Tuple[Dict[str, jnp.ndarray], RobotariumLogEnvState]:
        obs, env_state = self._env.reset(key)
        state = RobotariumLogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((self._env.num_agents,)),
            episode_lengths=jnp.zeros((self._env.num_agents,)),
            returned_episode_returns=jnp.zeros((self._env.num_agents,)),
            returned_episode_lengths=jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    def step(
        self,
        key: jax.Array,
        state: RobotariumLogEnvState,
        action: Union[int, float],
    ) -> Tuple[Dict[str, jnp.ndarray], RobotariumLogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = RobotariumLogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _copy_self_stream_to_partner_stream(two_stream_params):
    params = unfreeze(two_stream_params)
    target = params["params"] if "params" in params else params

    target["partner_mlp"] = target["self_mlp"]
    target["partner_ln"] = target["self_ln"]

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
    finetune_partner_stream = config.get("FINETUNE_OTHER_STREAM", False)

    if finetune_self_stream:
        trainable_paths.update(
            {
                ("params", "self_mlp"),
                ("params", "self_ln"),
            }
        )
    if finetune_partner_stream:
        trainable_paths.update(
            {
                ("params", "partner_mlp"),
                ("params", "partner_ln"),
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


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    assert env.num_agents == 3, "LMPred Foraging script currently supports 3 agents."

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    assert config["NUM_UPDATES"] > 0, "TOTAL_TIMESTEPS is too small for one update."
    assert (
        config["NUM_ACTORS"] * config["NUM_STEPS"]
    ) % config["NUM_MINIBATCHES"] == 0, "NUM_MINIBATCHES must divide rollout batch size."

    env = RobotariumLogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = config["LR"]
        lr_warmup = config["LR_WARMUP"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)
        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch,
        )
        return optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )

    def train(rng, seed_idx):
        ego_network = TwoStreamForagingActorCriticRNN(
            env.action_space(env.agents[0]).n, config=config
        )

        rng, _rng_reset, _rng_init = jax.random.split(rng, 3)
        reset_rng = jax.random.split(_rng_reset, config["NUM_ENVS"])
        obsv_init, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        obs0_init = obsv_init[env.agents[0]]
        init_x = (
            obs0_init[jnp.newaxis, ...],
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
        )

        fusion_hidden_dim = 3 * config["GRU_HIDDEN_DIM"]
        ego_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], fusion_hidden_dim
        )
        network_params = ego_network.init(_rng_init, ego_init_hstate, init_x)
        network_params = _copy_self_stream_to_partner_stream(network_params)

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

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ego_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], fusion_hidden_dim
        )

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    ego_hstate,
                    rng,
                ) = runner_state

                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.stack([last_obs[a] for a in env.agents])
                obs_shape = obs_batch.shape[2:]
                obs_batch = obs_batch.reshape(-1, *obs_shape)
                ac_in = (
                    obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                )

                ego_hstate, pi, value, _ = ego_network.apply(
                    train_state.params, ego_hstate, ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    ego_hstate,
                    rng,
                )
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_state, env_state, last_obs, last_done, update_step, ego_hstate, rng = (
                runner_state
            )
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents])
            obs_shape = last_obs_batch.shape[2:]
            last_obs_batch = last_obs_batch.reshape(-1, *obs_shape)
            ac_in = (
                last_obs_batch[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
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

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value, aux = ego_network.apply(
                            params,
                            jax.tree_util.tree_map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

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
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
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
            train_state = train_state.replace(
                params=_copy_self_stream_to_partner_stream(train_state.params)
            )
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

            def save_checkpoint(step_scalar, params, seed_scalar):
                current_step = int(step_scalar)
                seed_id = int(seed_scalar)
                num_skill_levels = config.get("NUM_SKILL_LEVELS", 10)
                save_interval = max(1, config["NUM_UPDATES"] // num_skill_levels)

                is_first_step = current_step <= 1
                is_interval_step = current_step % save_interval == 0
                if not (is_first_step or is_interval_step):
                    return

                checkpoints_prefix = config.get(
                    "CHECKPOINTS_PREFIX", "checkpoints/foraging/lmpred"
                )
                save_dir = checkpoints_prefix
                os.makedirs(save_dir, exist_ok=True)

                single_seed_params = jax.tree_util.tree_map(lambda x: np.array(x), params)
                bytes_data = flax.serialization.to_bytes(single_seed_params)
                file_path = os.path.join(
                    save_dir,
                    f"baseline_seed_{seed_id}_step_{current_step}.msgpack",
                )
                with open(file_path, "wb") as f:
                    f.write(bytes_data)
                print(
                    f"--> Saved seed {seed_id} checkpoint at step {current_step} to {file_path}"
                )

            jax.debug.callback(save_checkpoint, update_step, train_state.params, seed_idx)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                ego_hstate,
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
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def _parse_self_pred_gammas(self_pred_gammas):
    if isinstance(self_pred_gammas, str):
        gamma_str = self_pred_gammas.strip()
        if gamma_str.startswith("[") and gamma_str.endswith("]"):
            gamma_str = gamma_str[1:-1]
        if not gamma_str:
            return ()
        return tuple(float(gamma.strip()) for gamma in gamma_str.split(","))

    try:
        return tuple(float(gamma) for gamma in self_pred_gammas)
    except TypeError:
        return (float(self_pred_gammas),)


@hydra.main(version_base=None, config_path="", config_name="")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    num_seeds = config.get("NUM_SEEDS", 5)
    model_name = "lmpred"
    inferred_suffixes = []

    self_pred_coef = float(config.get("SELF_PRED_COEF", 0.1))
    if np.isclose(self_pred_coef, 0.0):
        inferred_suffixes.append("no_self_pred")

    self_pred_gammas = _parse_self_pred_gammas(
        config.get("SELF_PRED_GAMMAS", (0.0, 0.5, 0.9))
    )
    config["SELF_PRED_GAMMAS"] = self_pred_gammas
    if len(self_pred_gammas) == 1:
        if np.isclose(self_pred_gammas[0], 0.0):
            inferred_suffixes.append("gamma0")
        elif np.isclose(self_pred_gammas[0], 0.9):
            inferred_suffixes.append("gamma09")

    wandb_run_suffix = config.get("WANDB_RUN_SUFFIX", "")
    if wandb_run_suffix and wandb_run_suffix not in inferred_suffixes:
        inferred_suffixes.append(wandb_run_suffix)
    for suffix in inferred_suffixes:
        model_name += f"_{suffix}"

    perspective_transform = config.get("PERSPECTIVE_TRANSFORM", True)
    model_name += "_cpt" if perspective_transform else "_sameinp"
    run_name = config.get("RUN_NAME", model_name)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "LMPred", "JaxRobotarium", "Foraging"],
        config=config,
        mode=config["WANDB_MODE"],
        name=run_name,
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, num_seeds)
    train_jit = jax.jit(make_train(config))
    seed_ids = jnp.arange(num_seeds)
    jax.vmap(train_jit)(rngs, seed_ids)


if __name__ == "__main__":
    main()
