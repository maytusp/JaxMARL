# Population-Based Training cross-play for OvercookedV2.
import copy
import functools
import os
import random
from typing import Any, Callable, Dict, NamedTuple, Sequence

import distrax
import flax.linen as nn
import flax.serialization
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from jaxmarl.environments import overcooked_v2_layouts
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
from jaxmarl.wrappers.baselines import LogWrapper, OvercookedV2LogWrapper
from omegaconf import OmegaConf


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
        rnn_state = jnp.where(resets[:, np.newaxis], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        assert x.ndim == 4, f"CNN expected (B,H,W,C), got {x.shape}"
        x = nn.Conv(128, (1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(128, (1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(8, (1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(16, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.output_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        return self.activation(x)


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        assert obs.ndim == 5, f"Expected obs (T,B,H,W,C), got {obs.shape}"
        h, w, c = obs.shape[-3:]
        flat_obs = obs.reshape(-1, h, w, c)

        embedding = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )(flat_obs)
        embedding = embedding.reshape(*obs.shape[:-3], -1)
        embedding = nn.LayerNorm()(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

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
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def _as_float(x):
    return float(x.item() if hasattr(x, "item") else x)


def _as_int(x):
    return int(x.item() if hasattr(x, "item") else x)


def _make_tx(config, hparams):
    tx = optax.chain(
        optax.clip_by_global_norm(hparams["MAX_GRAD_NORM"]),
        optax.adam(hparams["LR"], eps=1e-5),
    )
    return tx


def _make_train_state(apply_fn, params, config, hparams, opt_state=None, step=None):
    tx = _make_tx(config, hparams)
    if opt_state is None:
        return TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
    return TrainState(
        step=jnp.asarray(0 if step is None else step),
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
    )


def _save_checkpoint(config, params, population_idx, pbt_iter):
    layout = config["ENV_KWARGS"]["layout"]
    checkpoints_prefix = config.get("CHECKPOINTS_PREFIX", "checkpoints/pbt/")
    save_dir = os.path.join(checkpoints_prefix, layout)
    os.makedirs(save_dir, exist_ok=True)

    single_params = jax.tree_util.tree_map(lambda x: np.array(x), params)
    bytes_data = flax.serialization.to_bytes(single_params)
    file_path = os.path.join(
        save_dir, f"baseline_seed_{population_idx}_step_{pbt_iter}.msgpack"
    )
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    print(f"--> Saved PBT population member {population_idx} at step {pbt_iter} to {file_path}")


def _bounded_mutate_lam(value, factor):
    del factor
    eps = min((1.0 - value) / 2.0, value / 2.0)
    direction = -1.0 if random.randint(0, 1) == 0 else 1.0
    return float(np.clip(value + direction * eps, 1e-4, 0.9999))


def _mutate_hparams(config, source_hparams):
    hparams = copy.deepcopy(source_hparams)
    resample_prob = float(config.get("PBT_RESAMPLE_PROB", 0.33))
    factors = list(config.get("PBT_MUTATION_FACTORS", [0.75, 1.25]))
    keys = list(
        config.get(
            "PBT_HYPERPARAMS_TO_MUTATE",
            ["GAE_LAMBDA", "CLIP_EPS", "LR", "UPDATE_EPOCHS", "ENT_COEF", "VF_COEF"],
        )
    )

    mutations = {}
    for key in keys:
        if key not in hparams or random.random() >= resample_prob:
            continue
        old_value = hparams[key]
        factor = float(random.choice(factors))
        if key == "GAE_LAMBDA":
            new_value = _bounded_mutate_lam(float(old_value), factor)
        elif isinstance(old_value, int):
            new_value = max(int(round(old_value * factor)), 1)
        else:
            new_value = float(old_value) * factor
            if key in ("LR", "ENT_COEF", "VF_COEF", "CLIP_EPS"):
                new_value = max(new_value, 1e-8)
        hparams[key] = new_value
        mutations[key] = (old_value, new_value, factor)
    return hparams, mutations


def make_train_pair(config, hparams):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    config = copy.deepcopy(config)
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    partner_network = ActorCriticRNN(env.action_space(env.agents[1]).n, config=config)
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=max(float(config["REW_SHAPING_HORIZON"]), 1.0),
    )

    def train_pair(rng, learner_state, partner_params, learner_ppo_runs):
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        partner_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        last_done = jnp.zeros((config["NUM_ENVS"],), dtype=bool)
        start_env_step = learner_ppo_runs * config["NUM_STEPS"] * config["NUM_ENVS"]

        def _env_step(carry, unused):
            env_state, last_obs, last_done, hstate, partner_hstate, episode_return, rng = carry
            rng, _rng_ego, _rng_partner, _rng_step = jax.random.split(rng, 4)

            ego_obs = last_obs[env.agents[0]]
            ac_in = (ego_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
            hstate, pi, value = network.apply(learner_state.params, hstate, ac_in)
            ego_action = pi.sample(seed=_rng_ego).squeeze(0)
            log_prob = pi.log_prob(ego_action).squeeze(0)

            partner_obs = last_obs[env.agents[1]]

            def _partner_act(obs, done, h, rng):
                partner_in = (
                    obs[jnp.newaxis, jnp.newaxis, ...],
                    done[jnp.newaxis, jnp.newaxis],
                )
                next_h, partner_pi, _ = partner_network.apply(
                    partner_params, h[jnp.newaxis, :], partner_in
                )
                return next_h.squeeze(0), partner_pi.sample(seed=rng).squeeze()

            partner_rng = jax.random.split(_rng_partner, config["NUM_ENVS"])
            partner_hstate, partner_action = jax.vmap(_partner_act)(
                partner_obs, last_done, partner_hstate, partner_rng
            )

            rng_step = jax.random.split(_rng_step, config["NUM_ENVS"])
            env_act = {env.agents[0]: ego_action, env.agents[1]: partner_action}
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state, env_act
            )

            original_reward = reward[env.agents[0]]
            anneal_factor = rew_shaping_anneal(start_env_step)
            reward = jax.tree_util.tree_map(
                lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
            )

            shaped_reward = info["shaped_reward"][env.agents[0]]
            combined_reward = reward[env.agents[0]]
            done_batch = done["__all__"]
            completed_return = episode_return + original_reward
            episode_return = jnp.where(done_batch, 0.0, completed_return)

            step_info = {
                "shaped_reward": shaped_reward,
                "original_reward": original_reward,
                "combined_reward": combined_reward,
                "anneal_factor": jnp.full((config["NUM_ENVS"],), anneal_factor),
                "returned_episode_returns": info["returned_episode_returns"],
                "returned_episode_lengths": info["returned_episode_lengths"],
                "returned_episode": info["returned_episode"],
                "pbt_completed_return": completed_return,
            }
            transition = Transition(
                done_batch,
                ego_action,
                value.squeeze(0),
                reward[env.agents[0]],
                log_prob,
                ego_obs,
                step_info,
            )
            carry = (env_state, obsv, done_batch, hstate, partner_hstate, episode_return, rng)
            return carry, transition

        carry = (
            env_state,
            obsv,
            last_done,
            hstate,
            partner_hstate,
            jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32),
            rng,
        )
        initial_hstate = hstate
        carry, traj_batch = jax.lax.scan(_env_step, carry, None, config["NUM_STEPS"])
        env_state, last_obs, last_done, hstate, partner_hstate, episode_return, rng = carry

        ac_in = (last_obs[env.agents[0]][jnp.newaxis, :], last_done[jnp.newaxis, :])
        _, _, last_val = network.apply(learner_state.params, hstate, ac_in)
        last_val = last_val.squeeze(0)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                delta = (
                    transition.reward
                    + hparams["GAMMA"] * next_value * (1 - transition.done)
                    - transition.value
                )
                gae = (
                    delta
                    + hparams["GAMMA"]
                    * hparams["GAE_LAMBDA"]
                    * (1 - transition.done)
                    * gae
                )
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _loss_fn(params, init_hstate, traj_batch, gae, targets):
            _, pi, value = network.apply(
                params, init_hstate.squeeze(0), (traj_batch.obs, traj_batch.done)
            )
            log_prob = pi.log_prob(traj_batch.action)
            value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
            ).clip(-hparams["CLIP_EPS"], hparams["CLIP_EPS"])
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                jnp.clip(ratio, 1.0 - hparams["CLIP_EPS"], 1.0 + hparams["CLIP_EPS"])
                * gae
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
            entropy = pi.entropy().mean()
            total_loss = (
                loss_actor
                + hparams["VF_COEF"] * value_loss
                - hparams["ENT_COEF"] * entropy
            )
            return total_loss, (value_loss, loss_actor, entropy)

        def _update_epoch(update_state, unused):
            def _update_minibatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, _), grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
            batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
            permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )
            train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
            return (train_state, init_hstate.squeeze(), traj_batch, advantages, targets, rng), total_loss

        update_state = (learner_state, initial_hstate, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, hparams["UPDATE_EPOCHS"]
        )
        learner_state = update_state[0]
        metric = jax.tree_util.tree_map(lambda x: x.mean(), traj_batch.info)
        metric["total_loss"] = loss_info.mean()
        return rng, learner_state, metric

    return train_pair


def make_eval_pair(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    eval_steps = int(config.get("PBT_NUM_SELECTION_STEPS", config["ENV_KWARGS"]["max_steps"]))
    eval_episodes = int(config.get("PBT_NUM_SELECTION_GAMES", 10))

    def eval_pair(rng, params_i, params_j, reward_shaping_factor):
        num_envs = eval_episodes
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        h_i = ScannedRNN.initialize_carry(num_envs, config["GRU_HIDDEN_DIM"])
        h_j = ScannedRNN.initialize_carry(num_envs, config["GRU_HIDDEN_DIM"])
        done = jnp.zeros((num_envs,), dtype=bool)

        def _step(carry, unused):
            env_state, obsv, done, h_i, h_j, dense_return, sparse_return, lengths, alive, rng = carry
            rng, _rng_i, _rng_j, _rng_step = jax.random.split(rng, 4)
            in_i = (obsv[env.agents[0]][jnp.newaxis, :], done[jnp.newaxis, :])
            in_j = (obsv[env.agents[1]][jnp.newaxis, :], done[jnp.newaxis, :])
            h_i, pi_i, _ = network.apply(params_i, h_i, in_i)
            h_j, pi_j, _ = network.apply(params_j, h_j, in_j)
            action_i = pi_i.sample(seed=_rng_i).squeeze(0)
            action_j = pi_j.sample(seed=_rng_j).squeeze(0)
            rng_step = jax.random.split(_rng_step, num_envs)
            env_act = {env.agents[0]: action_i, env.agents[1]: action_j}
            obsv, env_state, reward, done_next, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state, env_act
            )
            sparse_step_reward = reward[env.agents[0]]
            dense_step_reward = (
                reward[env.agents[0]]
                + info["shaped_reward"][env.agents[0]] * reward_shaping_factor
            )
            active = alive.astype(jnp.float32)
            dense_return = dense_return + active * dense_step_reward
            sparse_return = sparse_return + active * sparse_step_reward
            lengths = lengths + active
            alive = jnp.logical_and(alive, jnp.logical_not(done_next["__all__"]))
            return (
                env_state,
                obsv,
                done_next["__all__"],
                h_i,
                h_j,
                dense_return,
                sparse_return,
                lengths,
                alive,
                rng,
            ), None

        carry = (
            env_state,
            obsv,
            done,
            h_i,
            h_j,
            jnp.zeros((num_envs,), dtype=jnp.float32),
            jnp.zeros((num_envs,), dtype=jnp.float32),
            jnp.zeros((num_envs,), dtype=jnp.float32),
            jnp.ones((num_envs,), dtype=bool),
            rng,
        )
        carry, _ = jax.lax.scan(_step, carry, None, eval_steps)
        dense_return = carry[5]
        sparse_return = carry[6]
        lengths = jnp.maximum(carry[7], 1.0)
        return {
            "dense_reward_per_step": jnp.sum(dense_return) / jnp.sum(lengths),
            "sparse_return": jnp.mean(sparse_return),
            "episode_length": jnp.mean(lengths),
        }

    return eval_pair


def make_initial_population(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng_reset, _rng_init = jax.random.split(rng, 3)
    reset_rng = jax.random.split(_rng_reset, config["NUM_ENVS"])
    obsv_init, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    init_x = (
        obsv_init[env.agents[0]][jnp.newaxis, ...],
        jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
    )
    init_hstate = ScannedRNN.initialize_carry(
        config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
    )
    population_size = int(config["PBT_POPULATION_SIZE"])
    init_rngs = jax.random.split(_rng_init, population_size)
    base_hparams = {
        "LR": float(config["LR"]),
        "GAE_LAMBDA": float(config["GAE_LAMBDA"]),
        "CLIP_EPS": float(config["CLIP_EPS"]),
        "ENT_COEF": float(config["ENT_COEF"]),
        "VF_COEF": float(config["VF_COEF"]),
        "GAMMA": float(config["GAMMA"]),
        "MAX_GRAD_NORM": float(config["MAX_GRAD_NORM"]),
        "UPDATE_EPOCHS": int(config["UPDATE_EPOCHS"]),
    }

    states = []
    hparams = []
    for init_rng in init_rngs:
        params = network.init(init_rng, init_hstate, init_x)
        member_hparams = copy.deepcopy(base_hparams)
        states.append(_make_train_state(network.apply, params, config, member_hparams))
        hparams.append(member_hparams)
    return rng, states, hparams


@hydra.main(version_base=None, config_path="", config_name="")
def main(config):
    config = OmegaConf.to_container(config)
    config.setdefault("CHECKPOINTS_PREFIX", "checkpoints/pbt/")
    config.setdefault("PBT_POPULATION_SIZE", 4)
    config.setdefault("PBT_RESAMPLE_PROB", 0.33)
    config.setdefault("PBT_MUTATION_FACTORS", [0.75, 1.25])
    config.setdefault(
        "PBT_HYPERPARAMS_TO_MUTATE",
        ["GAE_LAMBDA", "CLIP_EPS", "LR", "UPDATE_EPOCHS", "ENT_COEF", "VF_COEF"],
    )
    config.setdefault("PBT_NUM_ITER", 10)
    config.setdefault("PBT_NUM_SELECTION_GAMES", 10)
    config.setdefault("PBT_NUM_SELECTION_STEPS", config["ENV_KWARGS"]["max_steps"])

    population_size = int(config["PBT_POPULATION_SIZE"])
    config["PBT_ITER_PER_SELECTION"] = int(
        config.get("PBT_ITER_PER_SELECTION", population_size ** 2)
    )
    layout_name = config["ENV_KWARGS"]["layout"]
    model_name = "pbt"
    if config["ENV_KWARGS"].get("front_obs", False):
        model_name += "_obsfront"

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["PBT", "RNN", "OvercookedV2"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"{model_name}_{layout_name}",
    )

    rng, population_states, population_hparams = make_initial_population(config)
    ppo_runs = [0 for _ in range(population_size)]
    best_sparse_returns = [-np.inf for _ in range(population_size)]

    eval_pair = jax.jit(make_eval_pair(config))
    train_cache = {}

    for pbt_iter in range(1, int(config["PBT_NUM_ITER"]) + 1):
        print(f"\nPBT ITERATION NUM {pbt_iter}")
        selection_reward_shaping_factor = 0.0
        pairs_to_train = [
            (partner_idx, learner_idx)
            for partner_idx in range(population_size)
            for learner_idx in range(population_size)
        ]
        random.shuffle(pairs_to_train)
        pairs_to_train = pairs_to_train[: config["PBT_ITER_PER_SELECTION"]]

        for sel_iter, (partner_idx, learner_idx) in enumerate(pairs_to_train, start=1):
            hparams_key = tuple(sorted(population_hparams[learner_idx].items()))
            if hparams_key not in train_cache:
                train_cache[hparams_key] = jax.jit(
                    make_train_pair(config, population_hparams[learner_idx])
                )
            train_pair = train_cache[hparams_key]
            rng, _rng = jax.random.split(rng)
            rng, new_state, metric = train_pair(
                _rng,
                population_states[learner_idx],
                population_states[partner_idx].params,
                ppo_runs[learner_idx],
            )
            population_states[learner_idx] = new_state
            ppo_runs[learner_idx] += 1
            metric = jax.tree_util.tree_map(_as_float, metric)
            selection_reward_shaping_factor = metric["anneal_factor"]
            metric.update(
                {
                    "pbt_iter": pbt_iter,
                    "pbt_selection_iter": sel_iter,
                    "pbt_partner_idx": partner_idx,
                    "pbt_learner_idx": learner_idx,
                    "pbt_learner_ppo_runs": ppo_runs[learner_idx],
                }
            )
            metric.update(
                {
                    f"pbt_learner_{k}": v
                    for k, v in population_hparams[learner_idx].items()
                    if isinstance(v, (int, float))
                }
            )
            wandb.log(metric)
            print(
                f"Training agent {learner_idx} ({ppo_runs[learner_idx]}) "
                f"with agent {partner_idx} fixed "
                f"(pbt {pbt_iter}/{config['PBT_NUM_ITER']}, "
                f"sel {sel_iter}/{len(pairs_to_train)})"
            )

        avg_dense_scores = [[] for _ in range(population_size)]
        avg_sparse_scores = [[] for _ in range(population_size)]
        for i in range(population_size):
            for j in range(i, population_size):
                rng, _rng = jax.random.split(rng)
                eval_metric = eval_pair(
                    _rng,
                    population_states[i].params,
                    population_states[j].params,
                    selection_reward_shaping_factor,
                )
                dense = _as_float(eval_metric["dense_reward_per_step"])
                sparse = _as_float(eval_metric["sparse_return"])
                avg_dense_scores[i].append(dense)
                avg_sparse_scores[i].append(sparse)
                if j != i:
                    avg_dense_scores[j].append(dense)
                    avg_sparse_scores[j].append(sparse)
                wandb.log(
                    {
                        "pbt_iter": pbt_iter,
                        "pbt_eval_i": i,
                        "pbt_eval_j": j,
                        "pbt_eval_dense_reward_per_step": dense,
                        "pbt_eval_sparse_return": sparse,
                        "pbt_eval_episode_length": _as_float(eval_metric["episode_length"]),
                    }
                )
                print(f"Evaluated agent {i} and {j}: dense_per_step={dense:.4f}")

        mean_dense = np.array([np.mean(scores) for scores in avg_dense_scores])
        mean_sparse = np.array([np.mean(scores) for scores in avg_sparse_scores])
        best_idx = int(np.argmax(mean_dense))
        worst_idx = int(np.argmin(mean_dense))

        for i in range(population_size):
            if mean_sparse[i] > best_sparse_returns[i]:
                best_sparse_returns[i] = mean_sparse[i]
                _save_checkpoint(config, population_states[i].params, i, pbt_iter)
            wandb.log(
                {
                    "pbt_iter": pbt_iter,
                    f"pbt_agent_{i}_mean_dense_reward_per_step": float(mean_dense[i]),
                    f"pbt_agent_{i}_mean_sparse_return": float(mean_sparse[i]),
                    f"pbt_agent_{i}_ppo_runs": ppo_runs[i],
                }
            )

        log_data = {
            "pbt_iter": pbt_iter,
            "pbt_best_idx": best_idx,
            "pbt_worst_idx": worst_idx,
            "pbt_best_score": float(mean_dense[best_idx]),
            "pbt_worst_score": float(mean_dense[worst_idx]),
        }
        is_final_iter = pbt_iter == int(config["PBT_NUM_ITER"])
        if is_final_iter:
            log_data["pbt_final_exploit_skipped"] = 1
            wandb.log(log_data)
            print(
                f"Final PBT iteration complete; skipped exploit so final "
                f"population checkpoints are not duplicated by a last copy."
            )
        else:
            source_state = population_states[best_idx]
            source_hparams = population_hparams[best_idx]
            mutated_hparams, mutations = _mutate_hparams(config, source_hparams)
            population_states[worst_idx] = _make_train_state(
                source_state.apply_fn,
                source_state.params,
                config,
                mutated_hparams,
                opt_state=source_state.opt_state,
                step=source_state.step,
            )
            population_hparams[worst_idx] = mutated_hparams
            ppo_runs[worst_idx] = ppo_runs[best_idx]

            for key, (_, new_value, factor) in mutations.items():
                log_data[f"pbt_mutation_{key}"] = float(new_value)
                log_data[f"pbt_mutation_{key}_factor"] = float(factor)
            wandb.log(log_data)
            print(
                f"Overwrote worst model {worst_idx} ({mean_dense[worst_idx]:.4f}) "
                f"with best model {best_idx} ({mean_dense[best_idx]:.4f}); "
                f"mutations={mutations}"
            )

    for idx, state in enumerate(population_states):
        _save_checkpoint(config, state.params, idx, int(config["PBT_NUM_ITER"]))


if __name__ == "__main__":
    main()
