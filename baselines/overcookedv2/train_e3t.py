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

        partner_action_logits = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="partner_action_fc",
        )(embedding)
        partner_action_logits = nn.relu(partner_action_logits)
        partner_action_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="partner_action_out",
        )(partner_action_logits)

        if self.config.get("USE_PRED_ACTION_INPUT", True):
            pred_partner_action = jax.lax.stop_gradient(
                nn.softmax(partner_action_logits, axis=-1)
            )
            actor_embedding = jnp.concatenate(
                [embedding, pred_partner_action], axis=-1
            )
        else:
            actor_embedding = embedding

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_embedding)
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

        return hidden, pi, jnp.squeeze(critic, axis=-1), partner_action_logits
        

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    partner_action: jnp.ndarray
    actor_mask: jnp.ndarray
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


def _config_get(config, key, default):
    if key in config:
        return config[key]
    train_kwargs = config.get("TRAIN_KWARGS", {})
    return train_kwargs.get(key, default)


def make_train(config):
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
    random_partner_prob = _config_get(
        config,
        "E3T_RANDOM_POLICY_PROB",
        _config_get(config, "e3t_beta", config.get("RAND", 0.0)),
    )
    moa_coef = config.get("MOA_COEF", config.get("PARTNER_ACTION_COEF", 1.0))
    action_dim = env.action_space(env.agents[0]).n

    def partner_action_targets(action):
        action_by_agent = action.reshape((env.num_agents, config["NUM_ENVS"]))
        if env.num_agents == 2:
            partner_action = jnp.flip(action_by_agent, axis=0)
        else:
            partner_action = jnp.roll(action_by_agent, shift=-1, axis=0)
        return partner_action.reshape((config["NUM_ACTORS"],))

    def train(rng, seed_idx):

        # INIT NETWORK
        network = ActorCriticRNN(action_dim, config=config)

        rng, _rng_reset, _rng_init = jax.random.split(rng, 3)

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
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
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

                hstate, pi, value, _ = network.apply(
                    train_state.params, hstate, ac_in
                )
                policy_action = pi.sample(seed=_rng)

                rng, _rng_random_action, _rng_random_partner = jax.random.split(rng, 3)
                random_action = jax.random.randint(
                    _rng_random_action,
                    policy_action.shape,
                    minval=0,
                    maxval=action_dim,
                )
                use_random_partner = (
                    jax.random.uniform(
                        _rng_random_partner, (config["NUM_ENVS"],)
                    )
                    < random_partner_prob
                )
                action_by_agent = policy_action.reshape(
                    (env.num_agents, config["NUM_ENVS"])
                )
                random_by_agent = random_action.reshape(
                    (env.num_agents, config["NUM_ENVS"])
                )
                if env.num_agents == 2:
                    partner_policy_mask = jnp.stack(
                        [
                            jnp.zeros((config["NUM_ENVS"],), dtype=bool),
                            use_random_partner,
                        ]
                    )
                else:
                    partner_policy_mask = jnp.broadcast_to(
                        use_random_partner, action_by_agent.shape
                    )
                    partner_policy_mask = partner_policy_mask.at[0].set(False)
                action = jnp.where(
                    partner_policy_mask, random_by_agent, action_by_agent
                ).reshape((config["NUM_ACTORS"],))
                log_prob = pi.log_prob(action)
                actor_mask = jnp.logical_not(partner_policy_mask).reshape(
                    (config["NUM_ACTORS"],)
                )
                partner_action = partner_action_targets(action)
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

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    partner_action.squeeze(),
                    actor_mask.squeeze(),
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
            _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
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
                        _, pi, value, partner_action_logits = network.apply(
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
                        actor_mask = traj_batch.actor_mask.astype(jnp.float32)
                        actor_mask_count = jnp.maximum(actor_mask.sum(), 1.0)
                        gae_mean = (gae * actor_mask).sum() / actor_mask_count
                        gae_var = (
                            jnp.square(gae - gae_mean) * actor_mask
                        ).sum() / actor_mask_count
                        gae = (gae - gae_mean) / (jnp.sqrt(gae_var) + 1e-8)

                        log_ratio = log_prob - traj_batch.log_prob
                        log_ratio = jnp.where(
                            traj_batch.actor_mask,
                            log_ratio,
                            0.0,
                        )
                        ratio = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))
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
                        loss_actor = (loss_actor * actor_mask).sum() / actor_mask_count
                        entropy = pi.entropy().mean()
                        partner_action_loss = optax.softmax_cross_entropy_with_integer_labels(
                            partner_action_logits, traj_batch.partner_action
                        ).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + moa_coef * partner_action_loss
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            partner_action_loss,
                        )

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
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["loss_total"] = loss_info[0].mean()
            metric["value_loss"] = loss_info[1][0].mean()
            metric["actor_loss"] = loss_info[1][1].mean()
            metric["entropy"] = loss_info[1][2].mean()
            metric["partner_action_loss"] = loss_info[1][3].mean()
            metric["random_partner_prob"] = random_partner_prob
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

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    model_name = "e3t"
    if config["ENV_KWARGS"].get("front_obs", False):
        model_name += "_obsfront"
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
