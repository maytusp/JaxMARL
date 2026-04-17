import functools
import os
from typing import Any, Callable, Dict, NamedTuple, Sequence

import distrax
import flax
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import flax.serialization
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from omegaconf import OmegaConf


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
        current_batch_size = ins.shape[0]

        if current_rnn_type != "gru":
            raise ValueError("")

        hidden_dim = self.config["GRU_HIDDEN_DIM"]
        new_carry = self.initialize_carry(self.config, rnn_type="gru", batch_size=current_batch_size)
        rnn_state = jnp.where(resets[:, None], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=hidden_dim)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(config, rnn_type=None, batch_size=None):
        if batch_size is None:
            batch_size = config["NUM_ENVS"]
        hidden_dim = config["GRU_HIDDEN_DIM"]
        cell = nn.GRUCell(features=hidden_dim)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_dim))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = False):
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


class PerspectivePredictor(nn.Module):
    feature_dim: int
    hidden_dim: int
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, h):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(h)
        x = self.activation(x)
        x = nn.Dense(
            self.feature_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return x


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    rnn_type: str = None

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        h, w, c = obs.shape[-3:]
        flat_obs = obs.reshape(-1, h, w, c)
        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = embed_model(flat_obs)
        embedding = embedding.reshape(*obs.shape[:-3], -1)
        embedding = nn.LayerNorm()(embedding)

        hidden, embedding = ScannedRNN(config=self.config, rnn_type=self.rnn_type)(hidden, (embedding, dones))

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
    info: Dict[str, jnp.ndarray]


class LossInfo(NamedTuple):
    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    entropy: jnp.ndarray


def load_cluster_checkpoint_lists(results_npz_path, heldout_seed_ids, final_step=220):
    data = np.load(results_npz_path, allow_pickle=True)

    checkpoint_names = data["checkpoint_names"].tolist()
    cluster_ids = data["cluster_ids"]

    cluster_to_names = {}
    for ckpt_name, cid in zip(checkpoint_names, cluster_ids):
        cid = int(cid)
        cluster_to_names.setdefault(cid, []).append(str(ckpt_name))

    # held-out eval partners stay fixed
    eval_checkpoint_names = [
        f"baseline_seed_{p_idx}_step_{final_step}.msgpack"
        for p_idx in heldout_seed_ids
    ]

    return cluster_to_names, eval_checkpoint_names


def load_partner_pool(config, checkpoints_prefix, checkpoint_names, dummy_params):
    layout_name = config["ENV_KWARGS"]["layout"]
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
    return ScannedRNN.initialize_carry(config, rnn_type="gru", batch_size=batch_size)


def build_ego_network(env, config):
    network_ego = ActorCriticRNN(
        env.action_space(env.agents[0]).n,
        config=config,
        rnn_type="gru",
    )
    init_hstate_ego = ScannedRNN.initialize_carry(config, rnn_type="gru", batch_size=config["NUM_ENVS"])
    return network_ego, init_hstate_ego


def make_zsc_evaluator(config, eval_pop_params):
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = config.get("EVAL_NUM_ENVS", 128)

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    network_ego, _ = build_ego_network(env, eval_config)
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

            hstate_ego = init_ego_hidden_for_batch(eval_config, num_eval_envs)
            hstate_partner = ScannedRNN.initialize_carry(eval_config, rnn_type="gru", batch_size=num_eval_envs)
            done_batch = jnp.zeros((num_eval_envs,), dtype=bool)
            ep_return = jnp.zeros((num_eval_envs,), dtype=jnp.float32)

            def _step_fn(carry, _):
                obs, env_state, hstate_ego, hstate_partner, done_batch, ep_return, rng = carry
                rng, _rng_partner, _rng_step = jax.random.split(rng, 3)

                obs_ego = obs[env.agents[0]]
                obs_partner = obs[env.agents[1]]

                ac_in_ego = (obs_ego[jnp.newaxis, :], done_batch[jnp.newaxis, :])
                ac_in_partner = (obs_partner[jnp.newaxis, :], done_batch[jnp.newaxis, :])

                hstate_ego, pi_ego, _= network_ego.apply(
                    ego_params,
                    hstate_ego,
                    ac_in_ego,
                )
                hstate_partner, pi_partner, _ = network_partner.apply(
                    partner_params,
                    hstate_partner,
                    ac_in_partner,
                )

                action_ego = jnp.argmax(pi_ego.logits, axis=-1).squeeze(0)
                action_partner = jnp.argmax(pi_partner.logits, axis=-1).squeeze(0)

                env_act = {
                    env.agents[0]: action_ego,
                    env.agents[1]: action_partner,
                }

                step_rng = jax.random.split(_rng_step, num_eval_envs)
                next_obs, next_env_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    step_rng, env_state, env_act
                )

                new_done_batch = done["__all__"]
                reward_team = 0.5 * (reward[env.agents[0]] + reward[env.agents[1]])
                ep_return = ep_return + reward_team

                new_carry = (
                    next_obs,
                    next_env_state,
                    hstate_ego,
                    hstate_partner,
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
                done_batch,
                ep_return,
                rng,
            )

            final_carry, _ = jax.lax.scan(_step_fn, init_carry, None, length=num_eval_steps)
            ep_return = final_carry[5]
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
    assert env.num_agents == 2, "This script currently assumes two Overcooked agents."

    zsc_evaluator = make_zsc_evaluator(config, eval_pop_params)
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    env = OvercookedV2LogWrapper(env, replace_info=False)
    obs_shape = env.observation_space().shape

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

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=config["REW_SHAPING_HORIZON"],
    )

    def train(rng, pop_params):
        network_ego, init_hstate_ego = build_ego_network(env, config)
        network_partner = ActorCriticRNN(env.action_space(env.agents[1]).n, config=config, rnn_type="gru")

        rng, _rng_ego = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *obs_shape)),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
        )
        network_params_ego = network_ego.init(_rng_ego, init_hstate_ego, init_x)

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
        train_state_ego = TrainState.create(apply_fn=network_ego.apply, params=network_params_ego, tx=tx)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        hstate_ego_start = init_ego_hidden_for_batch(config, config["NUM_ENVS"])
        hstate_partner_start = ScannedRNN.initialize_carry(config, rnn_type="gru", batch_size=config["NUM_ENVS"])
        init_hstate_combined = (hstate_ego_start, hstate_partner_start)

        # random partner
        rng, _rng_partner_init = jax.random.split(rng)
        partner_idx_vec = jax.random.randint(
            _rng_partner_init,
            shape=(config["NUM_ENVS"],),
            minval=0,
            maxval=config["POP_SIZE"],
        )

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate_combined,
                    partner_idx_vec,
                    rng,
                ) = runner_state

                hstate_ego, hstate_partner = hstate_combined
                rng, _rng_ego, _rng_partner, _rng_step, _rng_resample = jax.random.split(rng, 5)

                obs_ego = last_obs[env.agents[0]]
                obs_partner = last_obs[env.agents[1]]

                # ego action
                ac_in_ego = (obs_ego[jnp.newaxis, :], last_done[jnp.newaxis, :])
                hstate_ego, pi_ego, value_ego = network_ego.apply(
                    train_state.params,
                    hstate_ego,
                    ac_in_ego,
                )
                action_ego = pi_ego.sample(seed=_rng_ego)
                log_prob_ego = pi_ego.log_prob(action_ego)

                def take_partner_leaf(x):
                    # x: (POP_SIZE, ...)
                    return x[partner_idx_vec]   # -> (NUM_ENVS, ...)

                sampled_partner_params = jax.tree_util.tree_map(take_partner_leaf, pop_params)


                # simpler: use vmap over env dimension
                partner_rngs = jax.random.split(_rng_partner, config["NUM_ENVS"])

                def apply_partner_env(params_e, h_e, obs_e, done_e, rng_e):
                    h_in = h_e[None, :]
                    obs_in = obs_e[None, None, ...]
                    done_in = done_e[None, None, ...]

                    h_next, pi_e, _ = network_partner.apply(
                        params_e,
                        h_in,
                        (obs_in, done_in),
                    )
                    action_e = pi_e.sample(seed=rng_e).squeeze()
                    return h_next.squeeze(0), action_e

                hstate_partner, action_partner = jax.vmap(apply_partner_env)(
                    sampled_partner_params,
                    hstate_partner,
                    obs_partner,
                    last_done,
                    partner_rngs,
                )

                env_act = {
                    env.agents[0]: action_ego.flatten(),
                    env.agents[1]: action_partner.flatten(),
                }

                rng_step = jax.random.split(_rng_step, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                original_reward_ego = reward[env.agents[0]]
                current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                shaped_reward_ego = info["shaped_reward"][env.agents[0]]
                combined_reward_ego = reward[env.agents[0]]

                info["shaped_reward"] = shaped_reward_ego
                info["original_reward"] = original_reward_ego
                info["anneal_factor"] = jnp.full_like(shaped_reward_ego, anneal_factor)
                info["combined_reward"] = combined_reward_ego

                done_batch = done["__all__"]

                transition = Transition(
                    done=done_batch,
                    action=action_ego.squeeze(),
                    value=value_ego.squeeze(),
                    reward=combined_reward_ego.squeeze(),
                    log_prob=log_prob_ego.squeeze(),
                    obs=obs_ego,
                    info=info,
                )

                # resample partner ONLY for finished episodes
                new_partner_idx = jax.random.randint(
                    _rng_resample,
                    shape=(config["NUM_ENVS"],),
                    minval=0,
                    maxval=config["POP_SIZE"],
                )
                partner_idx_vec = jnp.where(done_batch, new_partner_idx, partner_idx_vec)

                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    (hstate_ego, hstate_partner),
                    partner_idx_vec,
                    rng,
                )
                return runner_state, transition

            hstate_combined = runner_state[-3]
            hstate_ego_init, _ = hstate_combined
            initial_hstate = hstate_ego_init
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            train_state, env_state, last_obs, last_done, update_step, hstate, partner_idx_vec, rng = runner_state
            hstate_ego, _ = hstate
            last_obs_ego = last_obs[env.agents[0]]
            ac_in = (last_obs_ego[jnp.newaxis, :], last_done[jnp.newaxis, :])
            _, _, last_val = network_ego.apply(train_state.params, hstate_ego, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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
                        squeezed_hstate = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), init_hstate)
                        _, pi, value = network_ego.apply(
                            params,
                            squeezed_hstate,
                            (traj_batch.obs, traj_batch.done),
                        )

                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()


                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        aux = LossInfo(
                            total_loss=total_loss,
                            value_loss=value_loss,
                            actor_loss=loss_actor,
                            entropy=entropy,
                        )
                        return total_loss, aux

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (_, loss_info), grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_info

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), init_hstate)
                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, loss_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (
                    train_state,
                    jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), init_hstate),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (train_state, initial_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            rng = update_state[-1]

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), traj_batch.info)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["loss/total"] = loss_info.total_loss.mean()
            metric["loss/value"] = loss_info.value_loss.mean()
            metric["loss/actor"] = loss_info.actor_loss.mean()
            metric["loss/entropy"] = loss_info.entropy.mean()
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
                update_step % config["EVAL_EVERY"] == 0,
                _run_eval,
                _skip_eval,
                operand=(train_state.params, eval_rng),
            )
            metric = {**metric, **eval_metrics}

            def callback(metric):
                wandb.log(metric)

            jax.debug.callback(callback, metric)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
                partner_idx_vec,
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
            partner_idx_vec,
            _rng,
        )
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        final_train_state = runner_state[0]
        return {
            "runner_state": runner_state,
            "metrics": metric,
            "final_params": final_train_state.params,
        }

    return train

def save_cluster_br_checkpoints(config, final_params, cluster_id):
    layout_name = config["ENV_KWARGS"]["layout"]
    save_root = config.get("CHECKPOINTS_PREFIX", "./checkpoints/cluster_br")
    save_dir = os.path.join(save_root, layout_name)
    os.makedirs(save_dir, exist_ok=True)

    num_seeds = jax.tree_util.tree_leaves(final_params)[0].shape[0]

    for seed_id in range(num_seeds):
        single_seed_params = jax.tree_util.tree_map(
            lambda x: np.array(x[seed_id]),
            final_params,
        )
        bytes_data = flax.serialization.to_bytes(single_seed_params)

        file_path = os.path.join(
            save_dir,
            f"cluster_{cluster_id}_seed_{seed_id}.msgpack",
        )
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        print(f"Saved cluster {cluster_id} seed {seed_id} -> {file_path}")

@hydra.main(version_base=None, config_path="config/oc_extended", config_name="")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    config["EVAL_EVERY"] = config.get("EVAL_EVERY", 50)
    config["EVAL_NUM_ENVS"] = config.get("EVAL_NUM_ENVS", 128)
    config["EVAL_NUM_EPISODES"] = config.get("EVAL_NUM_EPISODES", 8)
    config["EVAL_NUM_STEPS"] = config.get("EVAL_NUM_STEPS", config["NUM_STEPS"])
    config["PERSPECTIVE_AUX_COEF"] = config.get("PERSPECTIVE_AUX_COEF", 1.0)
    config["PERSPECTIVE_PRED_HIDDEN_DIM"] = config.get(
        "PERSPECTIVE_PRED_HIDDEN_DIM", config["GRU_HIDDEN_DIM"]
    )
    config["RNN_TYPE"] = "gru"

    layout_name = config["ENV_KWARGS"]["layout"]
    config["XP_RESULTS_PATH"] = f"./results/xp_similarity/{layout_name}/xp_similarity_results.npz"
    num_seeds = config["NUM_SEEDS"]
    print(f"num_seeds {num_seeds}")
    model_name = "cluster_br"
    checkpoint_prefix_train = "checkpoints/sp_pools_train"
    checkpoint_prefix_eval = "checkpoints/sp_pools_eval"

    # load cluster assignment from eval_xp output
    results_npz_path = config["XP_RESULTS_PATH"]   # e.g. ./results/xp_similarity/<layout>/xp_similarity_results.npz
    heldout_seed_ids = [i for i in range(25)] # use 25 unseen eval partners
    final_step = config.get("FCP_FINAL_STEP", 220)

    cluster_to_train_ckpts, eval_ckpts = load_cluster_checkpoint_lists(
        results_npz_path,
        heldout_seed_ids=heldout_seed_ids,
        final_step=final_step,
    )
    print(f"cluster_to_train_ckpts {cluster_to_train_ckpts}")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "OvercookedV2", "FCP", "ClusterBR"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"{model_name}_overcooked_v2_{layout_name}",
    )

    env_temp = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network_partner = ActorCriticRNN(
        env_temp.action_space(env_temp.agents[1]).n,
        config=config,
        rnn_type="gru",
    )

    init_x = (
        jnp.zeros((1, config["NUM_ENVS"], *env_temp.observation_space().shape)),
        jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
    )
    init_hstate = ScannedRNN.initialize_carry(config, rnn_type="gru")
    dummy_params = network_partner.init(jax.random.PRNGKey(0), init_hstate, init_x)
    
    eval_pop_params = load_partner_pool(config, checkpoint_prefix_eval, eval_ckpts, dummy_params)

    all_cluster_outputs = {}

    for cluster_id in sorted(cluster_to_train_ckpts.keys()):
        cluster_ckpts = cluster_to_train_ckpts[cluster_id]
        print(f"\n===== TRAINING BR FOR CLUSTER {cluster_id} =====")
        print("Train partners:", cluster_ckpts)

        cluster_config = dict(config)
        cluster_config["FCP_TRAIN_CHECKPOINTS"] = cluster_ckpts
        cluster_config["FCP_EVAL_CHECKPOINTS"] = eval_ckpts
        cluster_config["POP_SIZE"] = len(cluster_ckpts)
        cluster_config["EVAL_POP_SIZE"] = len(eval_ckpts)
        cluster_config["CLUSTER_ID"] = int(cluster_id)

        train_pop_params = load_partner_pool(cluster_config, checkpoint_prefix_train, cluster_ckpts, dummy_params)

        with jax.disable_jit(False):
            rng = jax.random.PRNGKey(cluster_config["SEED"] + int(cluster_id) * 1000)
            rngs = jax.random.split(rng, num_seeds)

            train_jit = jax.jit(make_train(cluster_config, eval_pop_params))
            out = jax.vmap(train_jit, in_axes=(0, None))(rngs, train_pop_params)

            save_cluster_br_checkpoints(
                config=cluster_config,
                final_params=out["final_params"],
                cluster_id=cluster_id,
            )

            _ = run_final_zsc_eval(
                config=cluster_config,
                final_params=out["final_params"],
                eval_pop_params=eval_pop_params,
            )

        all_cluster_outputs[int(cluster_id)] = out

    return all_cluster_outputs

if __name__ == "__main__":
    main()
