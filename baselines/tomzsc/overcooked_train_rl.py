"""
Adapted for JaxMARL OvercookedV2.
Keeps the original Q-learning / VDN-style training structure as much as possible.
"""

import os
import re
import copy
import json
from functools import partial
from typing import Any

import chex
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from tqdm import tqdm
from jax_tqdm import scan_tqdm

import jaxmarl
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper

from baselines.tomzsc.utils.wrappers import load_params, save_params
from baselines.tomzsc.models import JaxMARLLSTM, JaxMARLLSTMGCRL
from baselines.tomzsc.overcooked_cross_play import get_transitions, get_batch_size


def scan_loop(f, s=1):
    def wrapper(*arg):
        arg = jax.tree.map(lambda x: jnp.reshape(x, (-1, s, *x.shape[1:])), arg)
        _, res = jax.lax.scan(lambda a, b: (a, jax.vmap(f)(*b)), 0, xs=arg)
        res = jax.tree.map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), res)
        return res
    return wrapper


def get_shape(x):
    return jax.tree.map(lambda y: y.shape, x)


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    rewards: chex.Array
    done: chex.Array
    avail_actions: chex.Array
    q_vals: chex.Array
    state: Any = None
    concepts: chex.Array = None


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    teammate_params: Any = None


def make_train(config, env):
    assert (config["NUM_ENVS"] * config["NUM_STEPS"]) % config["NUM_MINIBATCHES"] == 0, (
        "NUM_ENVS*NUM_STEPS must be divisible by NUM_MINIBATCHES"
    )

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    rew_shaping_horizon = config["REW_SHAPING_HORIZON"]
    if rew_shaping_horizon > 0:
        rew_shaping_horizon = (
            config["TOTAL_TIMESTEPS"] * rew_shaping_horizon
            if rew_shaping_horizon <= 1
            else rew_shaping_horizon
        )
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.0, end_value=0.0, transition_steps=rew_shaping_horizon
        )
    else:
        rew_shaping_anneal = lambda *args: 0.0

    obs_shape = env.observation_space().shape
    action_dim = env.action_space(env.agents[0]).n

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def get_all_valid_actions(num_envs: int):
        return {
            agent: jnp.ones((num_envs, action_dim), dtype=jnp.float32)
            for agent in env.agents
        }

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1.0 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
        rng_a, rng_e = jax.random.split(rng)

        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        def get_random_actions(rng_, val_action):
            return jax.random.choice(
                rng_,
                jnp.arange(val_action.shape[-1]),
                p=val_action / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return chosen_actions

    def train(inputs):
        rng, teammate_params, all_params = inputs

        if config["PECAN"]:
            n_clusters = int(max(config["CLUSTER_LABELS"])) + 1

        original_seed = rng[0]

        if config.get("RANDOM_START_STATE", False):
            raise NotImplementedError(
                "RANDOM_START_STATE is not supported yet for OvercookedV2 in this script."
            )

        teammate_network = JaxMARLLSTM(
            action_dim=action_dim,
            norm_type=config["NORM_TYPE"],
            norm_input=config["NORM_INPUT"],
            use_lstm="teammate" in config.get("USE_LSTM", []),
        )

        if config["PECAN"]:
            network = JaxMARLLSTMGCRL(
                action_dim=action_dim,
                num_concepts=n_clusters + 1,
                norm_type=config["NORM_TYPE"],
                norm_input=config["NORM_INPUT"],
                use_lstm="ego" in config.get("USE_LSTM", []),
            )
        else:
            network = JaxMARLLSTM(
                action_dim=action_dim,
                norm_type=config["NORM_TYPE"],
                norm_input=config["NORM_INPUT"],
                use_lstm="ego" in config.get("USE_LSTM", []),
            )

        def create_agent(rng_):
            rng_, rng_init, _ = jax.random.split(rng_, 3)

            ego_class = JaxMARLLSTMGCRL if config["PECAN"] else JaxMARLLSTM
            init_hs = ego_class.initialize_carry(network.hidden_size, 1)
            init_x = jnp.zeros((1, 1, *obs_shape))
            init_dones = jnp.zeros((1, 1), dtype=bool)

            network_variables = network.init(
                rng_init, init_hs, init_x, init_dones, train=False
            )

            lr_scheduler = optax.linear_schedule(
                config["LR"],
                config["LR"] * config.get("LR_DECAY_FINISH", 0),
                config["NUM_EPOCHS"]
                * config["NUM_MINIBATCHES"]
                * config["NUM_UPDATES"]
                * config.get("LR_DECAY_TIME", 1),
            )
            exp_scheduler = optax.exponential_decay(
                config["LR"],
                config["NUM_EPOCHS"]
                * config["NUM_MINIBATCHES"]
                * config["NUM_UPDATES"],
                config.get("LR_DECAY_FINISH", 0.01),
            )
            lr = {
                "LINEAR": lr_scheduler,
                "EXP": exp_scheduler,
                "NONE": config["LR"],
            }[config.get("LR_DECAY", "NONE")]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            return CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )

        rng, _rng, __rng = jax.random.split(rng, 3)
        train_state = jax.vmap(create_agent)(jax.random.split(_rng, env.num_agents))

        if config["PECAN"]:
            sample_teammate = make_sample_teammate(config["CLUSTER_LABELS"])
            agent0_ids = jax.random.randint(_rng, (config["NUM_ENVS"],), 0, n_clusters + 1)
            policy_weights = jax.vmap(sample_teammate)(
                agent0_ids, jax.random.split(__rng, agent0_ids.shape[0])
            )
        else:
            agent0_ids = jax.random.randint(
                _rng, (config["NUM_ENVS"],), 0, max(config["NUM TEAMMATES"], 1)
            )
            policy_weights = None

        teammate_params = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), teammate_params
        )
        train_state = train_state.replace(teammate_params=teammate_params)

        def reset_batched(rng_, num_envs):
            reset_rng = jax.random.split(rng_, num_envs)
            obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            return obs, env_state

        def step_batched(rng_, env_state, actions):
            step_rng = jax.random.split(rng_, config["NUM_ENVS"])
            return jax.vmap(env.step, in_axes=(0, 0, 0))(step_rng, env_state, actions)

        # ---------------- TRAIN LOOP ----------------

        def _update_step(runner_state, unused):
            train_state, save_state, expl_state, test_state, rng = runner_state

            def _step_env(carry, _):
                (hs, teammate_hs), last_obs, last_dones, agent0_ids, policy_weights, cluster_returns, env_state, rng = carry

                rng, rng_a, rng_s, rng_o = jax.random.split(rng, 4)

                _obs = batchify(last_obs)[:, None]  # (num_agents, 1, num_envs, *obs)
                jax.debug.print("obs shape: {}", _obs.shape)
                
                if config["PECAN"]:
                    ego_params = jax.tree.map(
                        lambda x: jnp.stack([x[config["EGO"][0]]] * 2),
                        {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    )
                    hs, (q_vals, concept_output) = jax.vmap(
                        network.apply, in_axes=(0, 0, 0, None, None)
                    )(
                        ego_params,
                        hs,
                        _obs,
                        last_dones[None],
                        False,
                    )
                    concept_output = concept_output[:, 0]
                    q_vals = q_vals[:, 0]
                else:
                    hs, q_vals = jax.vmap(
                        network.apply, in_axes=(0, 0, 0, None, None)
                    )(
                        {"params": train_state.params, "batch_stats": train_state.batch_stats},
                        hs,
                        _obs,
                        last_dones[None],
                        False,
                    )
                    q_vals = q_vals[:, 0]

                avail_actions = get_all_valid_actions(config["NUM_ENVS"])

                eps = eps_scheduler(train_state.n_updates.mean())
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                actions = unbatchify(actions)

                # teammate actions
                if train_state.teammate_params is not None and config["NUM TEAMMATES"] > 0:
                    teammate_params_local = jax.tree.map(
                        lambda x: jnp.swapaxes(x, 0, 1), train_state.teammate_params
                    )

                    teammate_hs, opp_q_vals = jax.vmap(
                        jax.vmap(teammate_network.apply, in_axes=(0, 0, 0, None, None)),
                        in_axes=(0, 0, None, None, None),
                    )(
                        teammate_params_local,
                        teammate_hs,
                        _obs,
                        last_dones[None],
                        False,
                    )
                    opp_q_vals = opp_q_vals.squeeze(2)

                    if config["PECAN"]:
                        opp_q_vals = jnp.concatenate([opp_q_vals, q_vals[None]], axis=0).swapaxes(1, 2)
                        opp_q_vals = (opp_q_vals * policy_weights.T[:, :, None, None]).sum(axis=0).swapaxes(0, 1)
                    else:
                        opp_q_vals = opp_q_vals[agent0_ids, :, jnp.arange(agent0_ids.shape[0])].swapaxes(0, 1)

                    _rngs = jax.random.split(rng_o, env.num_agents)
                    opp_actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                        _rngs, opp_q_vals, eps, batchify(avail_actions)
                    )
                    opp_actions = unbatchify(opp_actions)

                    for k in config["TEAMMATE"]:
                        actions[f"agent_{k}"] = opp_actions[f"agent_{k}"]

                new_obs, new_env_state, reward, new_done, info = step_batched(
                    rng_s, env_state, actions
                )

                # shared sparse reward: use first ego agent
                ego_agent_name = f"agent_{config['EGO'][0]}"
                sparse_reward = reward[ego_agent_name]

                if "shaped_reward" in info:
                    shaped_reward = sum(info["shaped_reward"][f"agent_{k}"] for k in config["EGO"])
                else:
                    shaped_reward = jnp.zeros_like(sparse_reward)

                anneal = rew_shaping_anneal(train_state.timesteps.mean())
                team_reward = sparse_reward + anneal * shaped_reward

                # randomize teammate id after end of episode
                rng, _rng, __rng, rng4 = jax.random.split(rng, 4)
                if config["NUM TEAMMATES"] > 0:
                    agent0_ids = jnp.where(
                        new_done["__all__"],
                        jax.random.randint(_rng, agent0_ids.shape, 0, config["NUM TEAMMATES"]),
                        agent0_ids,
                    )

                if config["PECAN"]:
                    cluster_return_weights = (1 / cluster_returns) ** 3
                    cluster_return_weights = cluster_return_weights / cluster_return_weights.sum()
                    new_agent0_ids = jax.random.choice(
                        rng4, n_clusters + 1, agent0_ids.shape, p=cluster_return_weights
                    )
                    agent0_ids = jnp.where(new_done["__all__"], new_agent0_ids, agent0_ids)

                    new_policy_weights = jax.vmap(sample_teammate)(
                        agent0_ids, jax.random.split(__rng, agent0_ids.shape[0])
                    )
                    policy_weights = jax.tree.map(
                        lambda x, y: jnp.where(
                            jnp.expand_dims(new_done["__all__"], range(1, x.ndim)), x, y
                        ),
                        new_policy_weights,
                        policy_weights,
                    )

                    if "returned_episode_returns" in info:
                        episode_ret = info["returned_episode_returns"]
                    else:
                        episode_ret = team_reward

                    delta_cluster_returns = jnp.zeros_like(cluster_returns)
                    delta_cluster_returns = delta_cluster_returns.at[agent0_ids].add(
                        new_done["__all__"] * episode_ret
                    )
                    delta_n = jnp.zeros_like(cluster_returns)
                    delta_n = delta_n.at[agent0_ids].add(new_done["__all__"])
                    avg_delta_cluster_returns = delta_cluster_returns / jnp.where(
                        delta_n > 0, delta_n, jnp.ones_like(delta_n)
                    )
                    update_factor = jnp.where(
                        delta_n > 0,
                        jnp.ones_like(delta_n) * 0.95,
                        jnp.ones_like(delta_n),
                    )
                    cluster_returns = cluster_returns * update_factor + avg_delta_cluster_returns * (1 - update_factor)

                metric_info = {
                    "sparse_reward": sparse_reward,
                    "shaped_reward": shaped_reward,
                    "reward_annealing": jnp.full_like(sparse_reward, anneal),
                    "epsilon": jnp.full_like(sparse_reward, eps),
                }
                if "returned_episode" in info:
                    metric_info["returned_episode"] = info["returned_episode"]
                if "returned_episode_returns" in info:
                    metric_info["returned_episode_returns"] = info["returned_episode_returns"]

                next_avail_actions = get_all_valid_actions(config["NUM_ENVS"])

                transition = Transition(
                    obs=last_obs,
                    action=actions,
                    rewards=config.get("REW_SCALE", 1.0) * team_reward,
                    done=new_done["__all__"],
                    avail_actions=next_avail_actions,
                    q_vals=unbatchify(q_vals),
                    state=getattr(env_state, "env_state", env_state),
                    concepts=agent0_ids if config["PECAN"] else None,
                )

                new_carry = (
                    (hs, teammate_hs),
                    new_obs,
                    new_done["__all__"],
                    agent0_ids,
                    policy_weights,
                    cluster_returns,
                    new_env_state,
                    rng,
                )
                return new_carry, (transition, metric_info)

            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            (hs, teammate_hs), last_obs, dones, agent0_ids, policy_weights, cluster_returns, env_state = expl_state

            _, network_output = jax.vmap(
                network.apply, in_axes=(0, 0, 0, None, None)
            )(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                hs,
                batchify(last_obs)[:, None],
                dones[None],
                False,
            )

            last_q = network_output[0] if config["PECAN"] else network_output
            last_q = last_q[:, 0]
            last_q = jnp.max(last_q, axis=-1)
            last_q = sum(last_q[k] for k in config["EGO"])

            def _compute_targets(last_q_, q_vals_, reward_, done_):
                def _get_target(lambda_returns_and_next_q, rew_q_done):
                    reward_t, q_t, done_t = rew_q_done
                    lambda_returns, next_q = lambda_returns_and_next_q
                    target_bootstrap = reward_t + config["GAMMA"] * (1 - done_t) * next_q
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (1 - done_t) * lambda_returns + done_t * reward_t
                    next_q = jnp.max(q_t, axis=-1)
                    next_q = sum(next_q[k] for k in config["EGO"])
                    return (lambda_returns, next_q), lambda_returns

                lambda_returns = reward_[-1] + config["GAMMA"] * (1 - done_[-1]) * last_q_
                last_q_inner = jnp.max(q_vals_[-1], axis=-1)
                last_q_inner = sum(last_q_inner[k] for k in config["EGO"])
                _, targets_ = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q_inner),
                    jax.tree.map(lambda x: x[:-1], (reward_, q_vals_, done_)),
                    reverse=True,
                )
                targets_ = jnp.concatenate((targets_, lambda_returns[jnp.newaxis]))
                return targets_

            if config["NUM_STEPS"] > 1:
                q_vals = batchify(transitions.q_vals)
                q_vals = jnp.swapaxes(q_vals, 0, 1)
                lambda_targets = _compute_targets(
                    last_q,
                    q_vals,
                    transitions.rewards,
                    transitions.done,
                ).swapaxes(0, 1)
            else:
                lambda_targets = (
                    transitions.rewards[-1, 0]
                    + (1 - transitions.done[-1, 0]) * config["GAMMA"] * last_q
                )

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):
                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    _obs = batchify(minibatch.obs) if isinstance(minibatch.obs, dict) else minibatch.obs
                    _actions = batchify(minibatch.action) if isinstance(minibatch.action, dict) else minibatch.action

                    network_class = JaxMARLLSTMGCRL if config["PECAN"] else JaxMARLLSTM
                    hidden = network_class.initialize_carry(
                        network.hidden_size, len(env.agents), target.shape[1]
                    )

                    def _loss_fn(params):
                        (_, network_outputs), updates = jax.vmap(
                            partial(network.apply, train=True, mutable=["batch_stats"]),
                            in_axes=(0, 0, 0, None),
                        )(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            hidden,
                            _obs,
                            minibatch.done,
                        )

                        if config["PECAN"]:
                            q_vals_local, concept_outputs = network_outputs
                        else:
                            q_vals_local = network_outputs

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals_local,
                            jnp.expand_dims(_actions, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        chosen_action_qvals = sum(chosen_action_qvals[k] for k in config["EGO"])

                        q_loss = jnp.mean(
                            (chosen_action_qvals - jax.lax.stop_gradient(target)) ** 2
                        )
                        loss = q_loss
                        metrics = {"qvals": chosen_action_qvals}

                        if config["PECAN"]:
                            concept_outputs = concept_outputs[config["EGO"][0]]
                            concept_loss = optax.losses.softmax_cross_entropy_with_integer_labels(
                                concept_outputs, minibatch.concepts
                            ).mean()
                            loss = loss + concept_loss
                            metrics["concept_loss"] = concept_loss
                            metrics["concepts"] = minibatch.concepts

                        metrics["loss"] = loss
                        return loss, (updates, jax.tree.map(jnp.mean, metrics))

                    (loss, (updates, metrics)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)

                    train_state = jax.vmap(lambda g, t: t.apply_gradients(grads=g))(grads, train_state)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )

                    if config["PECAN"]:
                        train_state = train_state.replace(
                            params=jax.tree.map(
                                lambda x: jnp.stack([x[config["EGO"][0]]] * 2, axis=0),
                                train_state.params,
                            )
                        )

                    return (train_state, rng), jax.tree.map(jnp.mean, metrics)

                def preprocess_transition(x, indices):
                    x = x.swapaxes(0, 1)[indices]
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    ).swapaxes(1, 2)
                    return x

                rng, _rng = jax.random.split(rng)
                indices = jax.random.permutation(_rng, lambda_targets.shape[0])

                minibatches = jax.tree.map(
                    lambda x: preprocess_transition(x, indices),
                    transitions,
                )
                targets = lambda_targets[indices]
                targets = targets.reshape(
                    config["NUM_MINIBATCHES"], -1, *targets.shape[1:]
                ).swapaxes(1, 2)

                rng, _rng = jax.random.split(rng)
                (train_state, rng), metrics = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )
                return (train_state, rng), jax.tree.map(jnp.mean, metrics)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), metrics = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )
            metrics = jax.tree.map(jnp.mean, metrics)

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics.update(
                {
                    "env_step": train_state.timesteps.mean(),
                    "update_steps": train_state.n_updates.mean(),
                    "grad_steps": train_state.grad_steps.mean(),
                }
            )
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    (train_state.n_updates % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0).all()
                    | (train_state.n_updates == config["NUM_UPDATES"] - 1).all(),
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})
                save_state = jax.lax.cond(
                    metrics["test_returned_episode_returns"] >= save_state[0],
                    lambda: (metrics["test_returned_episode_returns"], train_state),
                    lambda: save_state,
                )
            if config["WANDB_MODE"] != "disabled":
                log_every = config.get("LOG_EVERY", 50)

                def callback(metrics_, original_seed_):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics_.update(
                            {f"rng{int(original_seed_)}/{k}": v for k, v in metrics_.items()}
                        )
                    wandb.log(metrics_)

                should_log = (
                    (train_state.n_updates % log_every == 0).all()
                    | (train_state.n_updates == 1).all()
                    | (train_state.n_updates == config["NUM_UPDATES"]).all()
                )

                jax.lax.cond(
                    should_log,
                    lambda _: jax.debug.callback(callback, metrics, original_seed),
                    lambda _: None,
                    operand=None,
                )

            runner_state = (train_state, save_state, tuple(expl_state), test_state, rng)
            return runner_state, metrics

        def get_greedy_metrics(rng, train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            def _greedy_env_step(teammate_params_local, step_state, unused):
                env_state, last_obs, (hs, teammate_hs), last_done, agent0_ids, rng = step_state
                rng, key_s = jax.random.split(rng)

                _obs = batchify(last_obs)[:, None]
                hs, q_vals = jax.vmap(network.apply, in_axes=(0, 0, 0, None))(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    hs,
                    _obs,
                    last_done[None],
                )
                if config["PECAN"]:
                    q_vals, _ = q_vals
                q_vals = q_vals[:, 0]

                valid_actions = get_all_valid_actions(config["TEST_NUM_ENVS"])
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)

                if teammate_params_local is not None and config["NUM TEAMMATES"] > 0:
                    teammate_params_local = jax.tree.map(
                        lambda x: jnp.swapaxes(x, 0, 1), teammate_params_local
                    )
                    teammate_hs, opp_q_vals = jax.vmap(
                        jax.vmap(teammate_network.apply, in_axes=(0, 0, 0, None, None)),
                        in_axes=(0, 0, None, None, None),
                    )(
                        teammate_params_local,
                        teammate_hs,
                        _obs,
                        last_done[None],
                        False,
                    )
                    opp_q_vals = opp_q_vals.squeeze(2)
                    opp_q_vals = opp_q_vals[agent0_ids, :, jnp.arange(agent0_ids.shape[0])].swapaxes(0, 1)
                    opp_actions = unbatchify(
                        jax.vmap(get_greedy_actions)(opp_q_vals, batchify(valid_actions))
                    )
                    for k in config["TEAMMATE"]:
                        actions[f"agent_{k}"] = opp_actions[f"agent_{k}"]

                step_rng = jax.random.split(key_s, config["TEST_NUM_ENVS"])
                obs, env_state, rewards, dones, infos = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_rng, env_state, actions)

                step_state = (
                    env_state,
                    obs,
                    (hs, teammate_hs),
                    dones["__all__"],
                    agent0_ids,
                    rng,
                )
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = reset_batched(_rng, config["TEST_NUM_ENVS"])

            rng, _rng = jax.random.split(rng)
            ego_class = JaxMARLLSTMGCRL if config["PECAN"] else JaxMARLLSTM
            hs = ego_class.initialize_carry(
                network.hidden_size, len(env.agents), config["TEST_NUM_ENVS"]
            )
            teammate_hs = JaxMARLLSTM.initialize_carry(
                teammate_network.hidden_size,
                max(config.get("NUM TEAMMATES", 1), 1),
                len(env.agents),
                config["TEST_NUM_ENVS"],
            )
            dones = jnp.zeros((config["TEST_NUM_ENVS"],), dtype=bool)
            agent0_ids = jax.random.randint(
                _rng, (config["TEST_NUM_ENVS"],), 0, max(config["NUM TEAMMATES"], 1)
            )

            rng, _rng = jax.random.split(rng)
            step_state = (env_state, init_obs, (hs, teammate_hs), dones, agent0_ids, _rng)
            _, (_, _, infos) = jax.lax.scan(
                partial(_greedy_env_step, train_state.teammate_params),
                step_state,
                None,
                config["TEST_NUM_STEPS"],
            )

            if "returned_episode" in infos and "returned_episode_returns" in infos:
                returned_episode = infos["returned_episode"]
                returned_episode_returns = infos["returned_episode_returns"]
                metric = jnp.nanmean(
                    jnp.where(returned_episode, returned_episode_returns, jnp.nan)
                )
            else:
                metric = jnp.nan

            return {"returned_episode_returns": metric}

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        rng, _rng = jax.random.split(rng)
        obs, env_state = reset_batched(_rng, config["NUM_ENVS"])

        ego_class = JaxMARLLSTMGCRL if config["PECAN"] else JaxMARLLSTM
        hs = ego_class.initialize_carry(network.hidden_size, len(env.agents), config["NUM_ENVS"])
        teammate_hs = JaxMARLLSTM.initialize_carry(
            teammate_network.hidden_size,
            max(config.get("NUM TEAMMATES", 1), 1),
            len(env.agents),
            config["NUM_ENVS"],
        )
        dones = jnp.zeros((config["NUM_ENVS"],), dtype=bool)

        if config["PECAN"]:
            cluster_returns = jnp.ones(n_clusters + 1)
        else:
            cluster_returns = None

        expl_state = ((hs, teammate_hs), obs, dones, agent0_ids, policy_weights, cluster_returns, env_state)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, (0.0, train_state), expl_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            scan_tqdm(config["NUM_UPDATES"])(_update_step),
            runner_state,
            xs=jnp.arange(config["NUM_UPDATES"]),
            length=config["NUM_UPDATES"],
        )

        def sample_report(x, n_samples=10):
            x_ = x[1:-1]
            n_points = x_.shape[0]
            x_ = jnp.reshape(
                x_[: (n_points // n_samples) * n_samples],
                (n_samples, -1, *x_.shape[1:]),
            )
            x_ = jnp.mean(x_, axis=1)
            return jnp.concatenate([x[:1], x_, x[-1:]], axis=0)

        report_metrics = jax.tree.map(sample_report, metrics)
        return {"runner_state": runner_state, "metrics": report_metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    layout_name = config["ENV_KWARGS"]["layout"]
    env_name_full = f"{env_name}_{layout_name}"

    env = jaxmarl.make(env_name, **config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)
    return env, env_name_full


def single_run(config, seed_offset=0):
    alg_name = config.get("ALG_NAME", "pqn_vdn_cnn")
    env, env_name = env_from_config(copy.deepcopy(config))

    def tree_batchify(x: dict):
        return jax.tree.map(
            lambda *args: jnp.stack(args, axis=0),
            *[x[agent] for agent in env.agents],
        )

    def tree_unbatchify(x: jnp.ndarray):
        return {
            agent: jax.tree.map(lambda x_: x_[i], x)
            for i, agent in enumerate(env.agents)
        }

    if config.get("TEAMMATE PARAMS", None) is not None:
        config["NUM TEAMMATES"] = len(config["TEAMMATE PARAMS"])
        teammate_params = [tree_batchify(p) for p in config["TEAMMATE PARAMS"]]
        teammate_params = jax.tree.map(lambda *args: jnp.stack(args, axis=0), *teammate_params)
    else:
        teammate_params = None
        config["NUM TEAMMATES"] = 0

    if config.get("ALL PARAMS", None) is not None:
        all_params = [tree_batchify(p) for p in config["ALL PARAMS"]]
        all_params = jax.tree.map(lambda *args: jnp.stack(args, axis=0), *all_params)
    else:
        all_params = None

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
        config={k: v for k, v in config.items() if k not in {"TEACHER PARAMS", "TEAMMATE PARAMS", "HEURISTIC ACTOR", "ALL PARAMS"}},
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"] + seed_offset)

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    all_params = jax.tree.map(lambda x: jnp.broadcast_to(x, (config["NUM_SEEDS"], *x.shape)), all_params)
    teammate_params = jax.tree.map(lambda x: jnp.broadcast_to(x, (config["NUM_SEEDS"], *x.shape)), teammate_params)

    parallel_seeds = config["NUM_SEEDS"] # if (config.get("TEAMMATE_DIR") is None and config["NUM_SEEDS"] % 5 == 0) else 1

    def scan_loop_local(f, arg):
        s = min(config["NUM_SEEDS"], parallel_seeds)
        arg = jax.tree.map(lambda x: jnp.reshape(x, (-1, s, *x.shape[1:])), arg)
        _, res = jax.lax.scan(lambda a, b: (a, jax.vmap(f)(b)), 0, xs=arg)
        res = jax.tree.map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), res)
        return res

    def get_res(rng_pack):
        return scan_loop_local(make_train(config, env), rng_pack)

    outs = jax.block_until_ready(jax.jit(get_res)((rngs, teammate_params, all_params)))

    if config.get("SAVE_PATH", None) is not None:
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)

        config_to_save = {k: v for k, v in config.items() if k not in {"TEAMMATE PARAMS", "ALL PARAMS"}}
        os.makedirs(os.path.join(save_dir, f"{alg_name}_{env_name}"), exist_ok=True)
        OmegaConf.save(
            config_to_save,
            os.path.join(save_dir, f"{alg_name}_{env_name}/seed{config['SEED']}_config.yaml"),
        )

        for i, _ in enumerate(rngs):
            params = {
                "params": jax.tree.map(lambda x: x[i], model_state.params),
                "batch_stats": jax.tree.map(lambda x: x[i], model_state.batch_stats),
            }
            params = tree_unbatchify(params)
            save_path = os.path.join(
                save_dir,
                f"{alg_name}_{env_name}/seed{config['SEED']}_vmap{i + seed_offset}.safetensors",
            )
            save_params(params, save_path)


def convert_clusters_to_labels(clusters):
    total = sum([len(x) for x in clusters])
    ret = np.zeros(total)
    for i, x in enumerate(clusters):
        ret[np.array(x)] = i
    return ret


def make_sample_teammate(cluster_labels):
    n_clusters = int(max(cluster_labels))
    n_teachers = len(cluster_labels)
    cluster_filter = jnp.zeros((n_clusters + 1, n_teachers + 1))
    cluster_filter = cluster_filter.at[-1, -1].set(1)

    for c in range(n_clusters):
        for i, l in enumerate(cluster_labels):
            if c == l:
                cluster_filter = cluster_filter.at[c, i].set(1)

    def sample_teammate(cluster_id, key):
        weights = jax.random.uniform(key, (n_teachers + 1,)) * cluster_filter[cluster_id]
        weights = jnp.float32(weights == weights.max())
        return weights

    return sample_teammate


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    config = {**config, **config["alg"]}

    teammate_dir = config.get("TEAMMATE_DIR")
    if teammate_dir is not None:
        teammate_files = sorted(
            [f for f in os.listdir(teammate_dir) if re.match(r"seed\d+_vmap\d+\.safetensors", f)]
        )
        teammate_paths = [os.path.join(teammate_dir, f) for f in teammate_files]

        cluster_labels = config.get("CLUSTER_LABELS")
        if cluster_labels is None:
            cluster_labels = [0] * len(teammate_paths)
        elif isinstance(cluster_labels, str):
            with open(cluster_labels, "r") as f:
                cluster_labels = json.load(f)
        config["CLUSTER_LABELS"] = cluster_labels

        unique_labels = np.unique(cluster_labels)
        all_params = [load_params(p) for p in teammate_paths]

        if config["PECAN"]:
            teammate_params = [[load_params(p) for p in teammate_paths]]
            teammate_paths = [teammate_paths]
        else:
            teammate_params = [
                [load_params(p) for p, l_ in zip(teammate_paths, cluster_labels) if l_ == l]
                for l in unique_labels
            ]
            teammate_paths = [
                [p for p, l_ in zip(teammate_paths, cluster_labels) if l_ == l]
                for l in unique_labels
            ]

        config["ALL PARAMS"] = all_params
    else:
        teammate_params = teammate_paths = (None,)
        if any(config.get(k) for k in ["TEAMMATE", "CLUSTER_LABELS", "PECAN"]):
            raise ValueError(
                "Response-training configs TEAMMATE, CLUSTER_LABELS, and PECAN are only available when specifying a TEAMMATE_DIR"
            )

    env, _ = env_from_config(config)
    if set(config["EGO"] + config.get("TEAMMATE", [])) != set(range(env.num_agents)):
        raise ValueError("Agents must either be classified as EGO or TEAMMATE")
    if set(config["EGO"]).intersection(config["TEAMMATE"]) != set():
        raise ValueError("Agents cannot be classified as both EGO and TEAMMATE")

    for i, (par, pat) in tqdm(enumerate(zip(teammate_params, teammate_paths))):
        print("\n\n")
        run_config = {**config, **{"TEAMMATE PARAMS": par, "TEAMMATE FILES": pat}}
        single_run(run_config, seed_offset=i)


if __name__ == "__main__":
    main()