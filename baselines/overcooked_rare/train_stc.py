import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
from flax.core import unfreeze
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

from baselines.overcooked_rare.stc_synapeses import (
    STCSynapses,
    create_stc_mask,
    normalize_capture,
    stc_disabled_metrics,
    summarize_stc_mask,
    topkmean,
    compute_tags,
    with_stc_defaults,
)


class STCTrainState(TrainState):
    stc_mask: Any = None
    latent_state: Any = None
    stc_capture_mean: Any = None
    stc_capture_var: Any = None
    stc_capture_count: Any = None
    stc_metrics: Any = None


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


class LatentPredictionModel(nn.Module):
    action_dim: int
    latent_dim: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, obs, next_obs, action):
        assert obs.ndim == 4, f"LatentPredictionModel expected obs (B,H,W,C), got {obs.shape}"
        encoder = CNN(output_size=self.latent_dim, activation=self.activation)
        z = encoder(obs)
        z_next = encoder(next_obs)
        action_onehot = jax.nn.one_hot(action.astype(jnp.int32), self.action_dim)
        dyn_in = jnp.concatenate([z, action_onehot], axis=-1)
        z_hat_next = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(dyn_in)
        z_hat_next = self.activation(z_hat_next)
        z_hat_next = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(z_hat_next)
        pred_error = jnp.mean(jnp.square(jax.lax.stop_gradient(z_next) - z_hat_next), axis=-1)
        loss = pred_error.mean()
        return loss, pred_error


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


def format_rare_prob_suffix(config):
    rare_prob = float(config["ENV_KWARGS"].get("rare_recipe_prob", 0.05))
    return f"_rareprob{rare_prob:g}".replace(".", "p")


def make_train(config):
    config = with_stc_defaults(config)
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
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
        action_dim = env.action_space(env.agents[0]).n
        network = ActorCriticRNN(action_dim, config=config)
        latent_model = LatentPredictionModel(
            action_dim=action_dim,
            latent_dim=int(config["latent_dim"]),
            activation=nn.relu if config["ACTIVATION"] == "relu" else nn.tanh,
        )

        rng, _rng_reset, _rng_init, _rng_latent = jax.random.split(rng, 4)

        reset_rng = jax.random.split(_rng_reset, config["NUM_ENVS"])
        obsv_init, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        obs0_init = obsv_init[env.agents[0]]
        init_x = (
            obs0_init[jnp.newaxis, ...],  # (1, NUM_ENVS, H, W, C)
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
        )

        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )

        network_params = network.init(_rng_init, init_hstate, init_x)
        latent_params = latent_model.init(
            _rng_latent,
            obs0_init,
            obs0_init,
            jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32),
        )
        
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
        latent_state = TrainState.create(
            apply_fn=latent_model.apply,
            params=latent_params,
            tx=optax.adam(config["latent_lr"], eps=1e-5),
        )
        stc_metrics = stc_disabled_metrics()
        if config["enable_stc"]:
            stc_mask = create_stc_mask(network_params, config)
            stc_mask_state = jax.tree_util.tree_map(
                lambda selected: jnp.asarray(selected, dtype=jnp.bool_),
                unfreeze(stc_mask),
            )
            stc_summary = summarize_stc_mask(network_params, stc_mask)
            print(
                "STC enabled: "
                f"{stc_summary['num_selected_leaves']} leaves, "
                f"{stc_summary['num_selected_scalars']} scalar actor params selected"
            )
            print("STC selected parameter paths:")
            for path in stc_summary["selected_paths"]:
                print(f"  + {path}")
            print("STC excluded parameter paths:")
            for path in stc_summary["excluded_paths"]:
                print(f"  - {path}")
        else:
            stc_mask_state = jax.tree_util.tree_map(
                lambda p: jnp.asarray(False, dtype=jnp.bool_),
                unfreeze(create_stc_mask(network_params, config)),
            )

        train_state = STCTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            stc_mask=stc_mask_state,
            latent_state=latent_state,
            stc_capture_mean=jnp.asarray(0.0, dtype=jnp.float32),
            stc_capture_var=jnp.asarray(1.0, dtype=jnp.float32),
            stc_capture_count=jnp.asarray(0.0, dtype=jnp.float32),
            stc_metrics=stc_metrics,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
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
                    hstate,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                # obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                obs_batch = jnp.stack([last_obs[a] for a in env.agents])
                obs_shape = obs_batch.shape[2:]
                obs_batch = obs_batch.reshape(-1, *obs_shape)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                env_act = {k: v.flatten() for k, v in env_act.items()}

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

                if "recipe_ingredient_ids" in info:
                    recipe_ingredient_ids = info.pop("recipe_ingredient_ids")
                    recipe_base = jnp.asarray(16, dtype=jnp.int32)
                    recipe_powers = recipe_base ** jnp.arange(
                        recipe_ingredient_ids.shape[-1], dtype=jnp.int32
                    )
                    recipe_id = jnp.sum(
                        recipe_ingredient_ids.astype(jnp.int32) * recipe_powers,
                        axis=-1,
                    )
                    info["recipe_id"] = jnp.tile(recipe_id, env.num_agents)
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
                    hstate,
                    rng,
                )
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
            )
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents])
            obs_shape = last_obs_batch.shape[2:]
            last_obs_batch = last_obs_batch.reshape(-1, *obs_shape)
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
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

            next_obs = jnp.concatenate(
                [traj_batch.obs[1:], last_obs_batch[jnp.newaxis, :]], axis=0
            )

            def _latent_loss_fn(latent_params, obs, next_obs, action):
                flat_obs = obs.reshape((-1,) + obs.shape[2:])
                flat_next_obs = next_obs.reshape((-1,) + next_obs.shape[2:])
                flat_action = action.reshape((-1,))
                loss, pred_error = latent_model.apply(
                    latent_params, flat_obs, flat_next_obs, flat_action
                )
                return loss, pred_error.reshape((obs.shape[0], obs.shape[1]))

            latent_grad_fn = jax.value_and_grad(_latent_loss_fn, has_aux=True)
            (latent_loss, latent_pred_error), latent_grads = latent_grad_fn(
                train_state.latent_state.params,
                traj_batch.obs,
                next_obs,
                traj_batch.action,
            )
            latent_state = train_state.latent_state.apply_gradients(grads=latent_grads)
            rare_recipe_mask = traj_batch.info["is_rare_recipe"].astype(jnp.float32)
            common_recipe_mask = 1.0 - rare_recipe_mask
            latent_pred_error_rare = (
                (latent_pred_error * rare_recipe_mask).sum()
                / jnp.maximum(rare_recipe_mask.sum(), 1.0)
            )
            latent_pred_error_common = (
                (latent_pred_error * common_recipe_mask).sum()
                / jnp.maximum(common_recipe_mask.sum(), 1.0)
            )

            capture_surprise = topkmean(latent_pred_error, config)
            advantage_sum = advantages.sum(axis=0)
            capture_raw = capture_surprise * jax.nn.relu(advantage_sum)
            capture, capture_mean, capture_var, capture_count = normalize_capture(
                capture_raw,
                train_state.stc_capture_mean,
                train_state.stc_capture_var,
                train_state.stc_capture_count,
                config,
            )

            def _single_actor_eligibility(params, init_hstate_actor, obs, done, action, gae):
                def _eligibility_loss(p):
                    _, pi, _ = network.apply(
                        p,
                        init_hstate_actor[jnp.newaxis, :],
                        (obs[:, jnp.newaxis, ...], done[:, jnp.newaxis]),
                    )
                    log_prob = pi.log_prob(action[:, jnp.newaxis]).squeeze(axis=1)
                    return jnp.sum(log_prob * gae)

                return jax.grad(_eligibility_loss)(params)

            eligibility = jax.vmap(
                _single_actor_eligibility,
                in_axes=(None, 0, 1, 1, 1, 1),
            )(
                train_state.params,
                initial_hstate,
                traj_batch.obs,
                traj_batch.done,
                traj_batch.action,
                advantages,
            )
            tags = compute_tags(eligibility, train_state.stc_mask, config)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
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

                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
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
                    init_hstate.squeeze(),
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
            if config["enable_stc"]:
                new_params, stc_metrics = STCSynapses.apply_consolidation(
                    params_after_ppo=train_state.params,
                    eligibility=eligibility,
                    tags=tags,
                    capture=capture,
                    stc_mask=train_state.stc_mask,
                    config=config,
                )
                stc_metrics["stc/capture"] = capture.mean()
                stc_metrics["stc/capture_surprise"] = capture_surprise.mean()
                stc_metrics["stc/latent_pred_error"] = latent_pred_error.mean()
                stc_metrics["stc/latent_pred_error_rare"] = latent_pred_error_rare
                stc_metrics["stc/latent_pred_error_common"] = latent_pred_error_common
            else:
                new_params = train_state.params
                stc_metrics = stc_disabled_metrics()
                stc_metrics["stc/latent_pred_error"] = latent_pred_error.mean()
                stc_metrics["stc/latent_pred_error_rare"] = latent_pred_error_rare
                stc_metrics["stc/latent_pred_error_common"] = latent_pred_error_common
            train_state = train_state.replace(
                params=new_params,
                latent_state=latent_state,
                stc_capture_mean=capture_mean,
                stc_capture_var=capture_var,
                stc_capture_count=capture_count,
                stc_metrics=stc_metrics,
            )
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            def recipe_success_callback(metric, step_scalar):
                if "recipe_id" not in metric:
                    return
                recipe_id = np.asarray(metric["recipe_id"]).reshape(-1)
                returned = np.asarray(metric["returned_episode"]).reshape(-1).astype(bool)
                returns = np.asarray(metric["returned_episode_returns"]).reshape(-1)
                log_data = {"update_step": int(step_scalar)}
                for rid in np.unique(recipe_id[returned]):
                    mask = returned & (recipe_id == rid)
                    if mask.any():
                        log_data[f"recipe_success_rate/{int(rid)}"] = float(
                            (returns[mask] > 0).mean()
                        )
                if len(log_data) > 1:
                    wandb.log(log_data)

            update_step = update_step + 1
            rare_episode_mask = metric["returned_episode"] & metric["is_rare_recipe"]
            rare_episode_count = rare_episode_mask.sum()
            metric["rare_episode_return"] = (
                jnp.where(
                    rare_episode_mask,
                    metric["returned_episode_returns"],
                    0.0,
                ).sum()
                / jnp.maximum(rare_episode_count, 1)
            )
            metric["rare_episode_count"] = rare_episode_count
            jax.debug.callback(recipe_success_callback, metric, update_step)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["ppo/return"] = metric["returned_episode_returns"]
            metric.update(train_state.stc_metrics)
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
                hstate,
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
            init_hstate,
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
    config = with_stc_defaults(config)

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    model_name = "stc_single"
    if config["ENV_KWARGS"].get("front_obs", False):
        model_name += "_obsfront"
    model_name += format_rare_prob_suffix(config)
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
        train_jit = jax.jit(make_train(config))
        seed_ids = jnp.arange(num_seeds)
        out = jax.vmap(train_jit)(rngs, seed_ids)


if __name__ == "__main__":
    main()
