import importlib
import os
import re
from typing import Dict, List, Tuple

import flax
import hydra
import imageio
import jax
import jax.numpy as jnp
import jaxmarl
import matplotlib.pyplot as plt
import numpy as np
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
from omegaconf import OmegaConf


METHOD_MODULES = {
    "sp": "baselines.overcookedv2.train_sp",
    "bf_sp": "baselines.overcookedv2.train_bf_sp",
    "ph2_v2": "baselines.overcookedv2.train_ph2_v2",
    "ph2_v2_ablate": "baselines.overcookedv2.train_ph2_v2",
    "privz": "baselines.overcookedv2.train_privz",
    "ph2_sp": "baselines.overcookedv2.train_ph2_sp",
    "e3t": "baselines.overcookedv2.train_e3t",
    "e3tlm": "baselines.overcookedv2.train_e3tlm",
    "ph2sf": "baselines.overcookedv2.train_ph2_sf",
    "ph2sf_ablate": "baselines.overcookedv2.train_ph2_sf",
    "ph2v3": "baselines.overcookedv2.train_ph2_v3",
    "ph2v3_ablate": "baselines.overcookedv2.train_ph2_v3",
    "ph2v4": "baselines.overcookedv2.train_ph2_v4",
    "ph2v4_ablate": "baselines.overcookedv2.train_ph2_v4",
    "lmpred_pop": "baselines.overcookedv2.train_lmpred_pop",
    "lmpred_pop_ablate": "baselines.overcookedv2.train_lmpred_pop",
    "fcp": "baselines.overcookedv2.train_fcp",
    "mep_pool": "baselines.overcookedv2.train_mep",
    "mep_br": "baselines.overcookedv2.train_mep",
}

CHECKPOINT_RE = re.compile(r"baseline_seed_(?P<seed>\d+)_step_(?P<step>\d+)\.msgpack$")
TWO_STREAM_METHODS = {"ph2_v1", "ph2_v2", "ph2_v2_ablate", "ph2_sp", "dual", "dual_ablation", "e3tlm", "ph2sf", 
                      "ph2sf_ablate", "ph2v3", "ph2v3_ablate", "ph2v4", "ph2v4_ablate",
                      "lmpred_pop", "lmpred_pop_ablate"}
FUSION_HIDDEN_METHODS = {"ph2_v2", "ph2_v2_ablate", "dual", "dual_ablation", "e3tlm", "ph2sf", 
                         "ph2sf_ablate", "ph2v3", "ph2v3_ablate", "ph2v4", "ph2v4_ablate",
                         "lmpred_pop", "lmpred_pop_ablate"}
TUPLE_HIDDEN_METHODS = {"ph2_sp"}
PRIVZ_METHODS = {"privz"}


def get_method_module(method: str):
    method_key = method.lower()
    if method_key not in METHOD_MODULES:
        choices = ", ".join(sorted(METHOD_MODULES))
        raise ValueError(f"Unknown TRAINING_METHOD={method!r}. Choose one of: {choices}")
    return importlib.import_module(METHOD_MODULES[method_key])


def get_network_class(method: str, method_module):
    if method.lower() in TWO_STREAM_METHODS:
        return method_module.TwoStreamActorCriticRNN
    return method_module.ActorCriticRNN


def get_hidden_dim(config: Dict) -> int:
    if config.get("TRAINING_METHOD", "sp").lower() in FUSION_HIDDEN_METHODS:
        return 2 * config["GRU_HIDDEN_DIM"]
    return config["GRU_HIDDEN_DIM"]


def initialize_hstate(config: Dict, method_module, batch_size: int):
    method = config.get("TRAINING_METHOD", "sp").lower()
    if method in TUPLE_HIDDEN_METHODS:
        return (
            method_module.ScannedRNN.initialize_carry(batch_size, config["GRU_HIDDEN_DIM"]),
            method_module.ScannedRNN.initialize_carry(batch_size, config["GRU_HIDDEN_DIM"]),
        )
    return method_module.ScannedRNN.initialize_carry(batch_size, get_hidden_dim(config))


def uses_priv_z(config: Dict) -> bool:
    return config.get("TRAINING_METHOD", "sp").lower() in PRIVZ_METHODS


def checkpoint_dir(config: Dict) -> str:
    layout_name = config["ENV_KWARGS"]["layout"]
    checkpoints_prefix = config.get("CHECKPOINTS_PREFIX", "./fcp_pool")
    return os.path.join(checkpoints_prefix, layout_name)


def default_result_name(config: Dict) -> str:
    result_name = config.get("XP_RESULT_NAME", None)
    if result_name is not None:
        return str(result_name)
    ego_method = str(config.get("EGO_TRAINING_METHOD", config.get("TRAINING_METHOD", "sp")))
    partner_method = str(config.get("PARTNER_TRAINING_METHOD", config.get("PARTNER_METHOD", ego_method)))
    return f"{ego_method}_ego_vs_{partner_method}_partner"


def make_role_config(config: Dict, role: str) -> Dict:
    role_config = dict(config)
    prefix = role.upper()
    role_config["TRAINING_METHOD"] = role_config.get(
        f"{prefix}_TRAINING_METHOD",
        role_config.get(
            f"{prefix}_METHOD",
            role_config.get("TRAINING_METHOD", "sp"),
        ),
    )
    role_config["CHECKPOINTS_PREFIX"] = role_config.get(
        f"{prefix}_CHECKPOINTS_PREFIX",
        role_config.get("CHECKPOINTS_PREFIX", "./fcp_pool"),
    )

    checkpoints = role_config.get(
        f"{prefix}_XP_CHECKPOINTS",
        role_config.get(f"{prefix}_CHECKPOINTS", None),
    )
    if checkpoints is not None:
        role_config["XP_CHECKPOINTS"] = checkpoints

    seeds = role_config.get(
        f"{prefix}_XP_SEEDS",
        role_config.get(f"{prefix}_SEEDS", None),
    )
    if seeds is not None:
        role_config["XP_SEEDS"] = seeds

    latest_per_seed = role_config.get(f"{prefix}_XP_LATEST_PER_SEED", None)
    if latest_per_seed is not None:
        role_config["XP_LATEST_PER_SEED"] = latest_per_seed

    return role_config


def discover_checkpoints(config: Dict) -> List[str]:
    ckpt_dir = checkpoint_dir(config)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {ckpt_dir}. "
            "Set CHECKPOINTS_PREFIX to the method-specific training output directory."
        )

    names = []
    for name in os.listdir(ckpt_dir):
        if CHECKPOINT_RE.match(name):
            names.append(name)
    if not names:
        raise FileNotFoundError(f"No baseline_seed_*_step_*.msgpack files found in {ckpt_dir}")

    names = sorted(names, key=checkpoint_sort_key)
    step_filter = config.get("XP_CHECKPOINT_STEP", None)
    if step_filter not in (None, ""):
        step_filter = int(step_filter)
        names = [
            name
            for name in names
            if int(CHECKPOINT_RE.match(name).group("step")) == step_filter
        ]
        if not names:
            raise ValueError(
                f"No checkpoints found at XP_CHECKPOINT_STEP={step_filter} in {ckpt_dir}."
            )

    if config.get("XP_LATEST_PER_SEED", True):
        latest_by_seed = {}
        for name in names:
            match = CHECKPOINT_RE.match(name)
            seed = int(match.group("seed"))
            step = int(match.group("step"))
            if seed not in latest_by_seed or step > latest_by_seed[seed][0]:
                latest_by_seed[seed] = (step, name)
        names = [latest_by_seed[seed][1] for seed in sorted(latest_by_seed)]

    seed_filter = config.get("XP_SEEDS", None)
    if seed_filter is not None:
        seed_filter = {int(seed) for seed in seed_filter}
        names = [
            name
            for name in names
            if int(CHECKPOINT_RE.match(name).group("seed")) in seed_filter
        ]

    if not names:
        raise ValueError("No checkpoints remain after applying XP_SEEDS.")
    return names


def checkpoint_sort_key(name: str) -> Tuple[int, int, str]:
    match = CHECKPOINT_RE.match(name)
    if match is None:
        return (10**12, 10**12, name)
    return (int(match.group("seed")), int(match.group("step")), name)


def make_network_and_dummy_params(config: Dict, method_module):
    eval_config = dict(config)
    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    network_cls = get_network_class(eval_config.get("TRAINING_METHOD", "sp"), method_module)
    network = network_cls(
        env.action_space(env.agents[0]).n,
        config=eval_config,
    )

    rng = jax.random.PRNGKey(eval_config.get("SEED", 0))
    rng, reset_rng, init_rng = jax.random.split(rng, 3)
    reset_rng = jax.random.split(reset_rng, get_eval_num_episodes(eval_config, default=1))
    obsv_init, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

    batch_size = get_eval_num_episodes(eval_config, default=1)
    hstate = initialize_hstate(eval_config, method_module, batch_size)
    init_x = [
        obsv_init[env.agents[0]][jnp.newaxis, ...],
        jnp.zeros((1, batch_size), dtype=bool),
    ]
    if uses_priv_z(eval_config):
        init_x.append(jnp.zeros((1, batch_size, eval_config["GRU_HIDDEN_DIM"]), dtype=jnp.float32))
    init_x = tuple(init_x)
    dummy_params = network.init(init_rng, hstate, init_x)
    return network, dummy_params


def load_agent_pool(config: Dict, method_module):
    names = config.get("XP_CHECKPOINTS", None)
    if names is None:
        names = discover_checkpoints(config)
    elif isinstance(names, str):
        names = [names]
    else:
        names = list(names)

    _, dummy_params = make_network_and_dummy_params(config, method_module)
    ckpt_dir = checkpoint_dir(config)
    loaded_params = []
    for name in names:
        ckpt_path = name if os.path.isabs(name) else os.path.join(ckpt_dir, name)
        with open(ckpt_path, "rb") as f:
            loaded_params.append(flax.serialization.from_bytes(dummy_params, f.read()))
        print(f"LOADED: {ckpt_path}")

    stacked_params = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *loaded_params)
    return {
        "params": stacked_params,
        "names": names,
    }


def get_pool_params_i(pool_params, i: int):
    return jax.tree_util.tree_map(lambda x: x[i], pool_params)


def get_pool_params_batch(pool_params, indices: np.ndarray):
    return jax.tree_util.tree_map(lambda x: x[indices], pool_params)


def get_eval_num_episodes(config: Dict, default: int = 128) -> int:
    return int(config.get("EVAL_NUM_EPISODES", config.get("EVAL_NUM_ENVS", default)))


def get_alignment_num_episodes(config: Dict, default: int = 128) -> int:
    return int(
        config.get(
            "ALIGN_EVAL_NUM_EPISODES",
            config.get(
                "ALIGN_EVAL_NUM_ENVS",
                config.get("EVAL_NUM_EPISODES", config.get("EVAL_NUM_ENVS", default)),
            ),
        )
    )


def batched_pairs(num_ego_agents: int, num_partner_agents: int, batch_size: int):
    pairs = [(i, j) for i in range(num_ego_agents) for j in range(num_partner_agents)]
    batch_size = max(1, int(batch_size))
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        valid_count = len(batch)
        if valid_count < batch_size:
            batch = batch + [batch[-1]] * (batch_size - valid_count)
        yield batch, valid_count


def make_xp_evaluator(
    config: Dict,
    ego_config: Dict,
    partner_config: Dict,
    ego_method_module,
    partner_method_module,
    ego_pool: Dict,
    partner_pool: Dict,
):
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = get_eval_num_episodes(eval_config, default=128)

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    ego_network_cls = get_network_class(ego_config.get("TRAINING_METHOD", "sp"), ego_method_module)
    ego_network = ego_network_cls(
        env.action_space(env.agents[0]).n,
        config=ego_config,
    )
    partner_network_cls = get_network_class(
        partner_config.get("TRAINING_METHOD", "sp"),
        partner_method_module,
    )
    partner_network = partner_network_cls(
        env.action_space(env.agents[0]).n,
        config=partner_config,
    )

    ego_pool_params = ego_pool["params"]
    partner_pool_params = partner_pool["params"]
    num_ego_agents = jax.tree_util.tree_leaves(ego_pool_params)[0].shape[0]
    num_partner_agents = jax.tree_util.tree_leaves(partner_pool_params)[0].shape[0]
    num_eval_steps = eval_config.get(
        "EVAL_NUM_STEPS",
        eval_config["ENV_KWARGS"].get("max_steps", 400),
    )
    num_eval_episodes = eval_config["NUM_ENVS"]
    sample_actions = eval_config.get("EVAL_SAMPLE_ACTIONS", False)

    def apply_policy(network, policy_config, params, hstate, obs, done_batch, rng, priv_z=None):
        ac_in = [obs[jnp.newaxis, :], done_batch[jnp.newaxis, :]]
        if uses_priv_z(policy_config):
            ac_in.append(priv_z[jnp.newaxis, :])
        ac_in = tuple(ac_in)
        out = network.apply(params, hstate, ac_in)
        hstate, pi = out[0], out[1]
        if sample_actions:
            action = pi.sample(seed=rng).squeeze(0)
        else:
            action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
        return hstate, action

    def evaluate_one_order(rng, params_a, params_b):
        rng, reset_key = jax.random.split(rng)
        reset_rng = jax.random.split(reset_key, num_eval_episodes)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        hstate_a = initialize_hstate(ego_config, ego_method_module, num_eval_episodes)
        hstate_b = initialize_hstate(partner_config, partner_method_module, num_eval_episodes)
        done_batch = jnp.zeros((num_eval_episodes,), dtype=bool)
        ep_return = jnp.zeros((num_eval_episodes,), dtype=jnp.float32)

        def step_fn(carry, _):
            obs, env_state, hstate_a, hstate_b, done_batch, ep_return, rng = carry
            rng, rng_a, rng_b, rng_step = jax.random.split(rng, 4)
            old_hstate_a = hstate_a
            old_hstate_b = hstate_b

            hstate_a, action_a = apply_policy(
                ego_network,
                ego_config,
                params_a,
                hstate_a,
                obs[env.agents[0]],
                done_batch,
                rng_a,
                priv_z=old_hstate_b,
            )
            hstate_b, action_b = apply_policy(
                partner_network,
                partner_config,
                params_b,
                hstate_b,
                obs[env.agents[1]],
                done_batch,
                rng_b,
                priv_z=old_hstate_a,
            )

            env_act = {
                env.agents[0]: action_a,
                env.agents[1]: action_b,
            }
            step_rng = jax.random.split(rng_step, num_eval_episodes)
            next_obs, next_env_state, reward, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_rng, env_state, env_act)

            team_reward = 0.5 * (reward[env.agents[0]] + reward[env.agents[1]])
            ep_return = ep_return + team_reward
            return (
                next_obs,
                next_env_state,
                hstate_a,
                hstate_b,
                done["__all__"],
                ep_return,
                rng,
            ), None

        init_carry = (obsv, env_state, hstate_a, hstate_b, done_batch, ep_return, rng)
        final_carry, _ = jax.lax.scan(step_fn, init_carry, None, length=num_eval_steps)
        return final_carry[5].mean()

    evaluate_pair_batch_jit = jax.jit(jax.vmap(evaluate_one_order, in_axes=(0, 0, 0)))

    def evaluator(rng):
        xp_matrix = np.zeros((num_ego_agents, num_partner_agents), dtype=np.float32)
        pair_batch_size = int(config.get("XP_PAIR_BATCH_SIZE", 8))
        print(
            "Using XP rollout scale: "
            f"XP_PAIR_BATCH_SIZE={pair_batch_size}, "
            f"EVAL_NUM_EPISODES={num_eval_episodes}, "
            f"EVAL_NUM_STEPS={num_eval_steps}"
        )

        for pair_batch, valid_count in batched_pairs(
            num_ego_agents,
            num_partner_agents,
            pair_batch_size,
        ):
            ego_indices = np.array([i for i, _ in pair_batch], dtype=np.int32)
            partner_indices = np.array([j for _, j in pair_batch], dtype=np.int32)
            params_i = get_pool_params_batch(ego_pool_params, ego_indices)
            params_j = get_pool_params_batch(partner_pool_params, partner_indices)
            rng, eval_rng = jax.random.split(rng)
            eval_rngs = jax.random.split(eval_rng, len(pair_batch))
            returns = np.asarray(evaluate_pair_batch_jit(eval_rngs, params_i, params_j))

            for batch_idx, (i, j) in enumerate(pair_batch[:valid_count]):
                ret_ij = float(returns[batch_idx])
                xp_matrix[i, j] = ret_ij
                print(f"XP ego[{i:02d}] partner[{j:02d}] = {ret_ij:.3f}")

        return {
            "xp_matrix": xp_matrix,
            "ego_checkpoint_names": list(ego_pool["names"]),
            "partner_checkpoint_names": list(partner_pool["names"]),
            "ego_training_method": ego_config.get("TRAINING_METHOD", "sp"),
            "partner_training_method": partner_config.get("TRAINING_METHOD", "sp"),
            "result_name": default_result_name(config),
            "ego_checkpoints_prefix": ego_config.get("CHECKPOINTS_PREFIX", "./fcp_pool"),
            "partner_checkpoints_prefix": partner_config.get("CHECKPOINTS_PREFIX", "./fcp_pool"),
            "layout": config["ENV_KWARGS"]["layout"],
        }

    return evaluator


def hidden_to_array(hidden):
    if isinstance(hidden, tuple):
        return jnp.concatenate([hidden_to_array(h) for h in hidden], axis=-1)
    return hidden


def make_partner_alignment_collector(
    config: Dict,
    ego_config: Dict,
    partner_config: Dict,
    ego_method_module,
    partner_method_module,
):
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = get_alignment_num_episodes(eval_config, default=128)

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    ego_network_cls = get_network_class(ego_config.get("TRAINING_METHOD", "sp"), ego_method_module)
    ego_network = ego_network_cls(
        env.action_space(env.agents[0]).n,
        config=ego_config,
    )
    partner_network_cls = get_network_class(
        partner_config.get("TRAINING_METHOD", "sp"),
        partner_method_module,
    )
    partner_network = partner_network_cls(
        env.action_space(env.agents[0]).n,
        config=partner_config,
    )

    num_eval_steps = eval_config.get(
        "ALIGN_EVAL_NUM_STEPS",
        eval_config.get(
            "EVAL_NUM_STEPS",
            eval_config["ENV_KWARGS"].get("max_steps", 400),
        ),
    )
    num_eval_episodes = eval_config["NUM_ENVS"]
    sample_actions = eval_config.get("EVAL_SAMPLE_ACTIONS", False)

    def apply_policy(network, policy_config, params, hstate, obs, done_batch, rng, priv_z=None):
        ac_in = [obs[jnp.newaxis, :], done_batch[jnp.newaxis, :]]
        if uses_priv_z(policy_config):
            ac_in.append(priv_z[jnp.newaxis, :])
        out = network.apply(params, hstate, tuple(ac_in))
        hstate, pi = out[0], out[1]
        if sample_actions:
            action = pi.sample(seed=rng).squeeze(0)
        else:
            action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
        return hstate, action

    def collect_one_pair(rng, params_a, params_b):
        rng, reset_key = jax.random.split(rng)
        reset_rng = jax.random.split(reset_key, num_eval_episodes)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        hstate_a = initialize_hstate(ego_config, ego_method_module, num_eval_episodes)
        hstate_b = initialize_hstate(partner_config, partner_method_module, num_eval_episodes)
        done_batch = jnp.zeros((num_eval_episodes,), dtype=bool)

        def step_fn(carry, _):
            obs, env_state, hstate_a, hstate_b, done_batch, rng = carry
            rng, rng_a, rng_b, rng_step = jax.random.split(rng, 4)
            old_hstate_a = hstate_a
            old_hstate_b = hstate_b

            hstate_a, action_a = apply_policy(
                ego_network,
                ego_config,
                params_a,
                hstate_a,
                obs[env.agents[0]],
                done_batch,
                rng_a,
                priv_z=old_hstate_b,
            )
            hstate_b, action_b = apply_policy(
                partner_network,
                partner_config,
                params_b,
                hstate_b,
                obs[env.agents[1]],
                done_batch,
                rng_b,
                priv_z=old_hstate_a,
            )

            env_act = {
                env.agents[0]: action_a,
                env.agents[1]: action_b,
            }
            step_rng = jax.random.split(rng_step, num_eval_episodes)
            next_obs, next_env_state, _, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_rng, env_state, env_act)

            transition = {
                "z_ego": hidden_to_array(hstate_a),
                "z_partner": hidden_to_array(hstate_b),
                "done": done["__all__"],
            }
            next_carry = (
                next_obs,
                next_env_state,
                hstate_a,
                hstate_b,
                done["__all__"],
                rng,
            )
            return next_carry, transition

        init_carry = (obsv, env_state, hstate_a, hstate_b, done_batch, rng)
        _, traj = jax.lax.scan(step_fn, init_carry, None, length=num_eval_steps)
        return traj

    return jax.jit(collect_one_pair)


def episode_split(num_episodes: int, train_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    if num_episodes < 2:
        raise ValueError("Alignment evaluation requires ALIGN_EVAL_NUM_EPISODES >= 2.")
    train_count = int(np.floor(num_episodes * train_fraction))
    train_count = min(max(train_count, 1), num_episodes - 1)
    train_idx = np.arange(train_count)
    test_idx = np.arange(train_count, num_episodes)
    return train_idx, test_idx


def flatten_episodes(z: np.ndarray, episode_idx: np.ndarray) -> np.ndarray:
    selected = z[:, episode_idx]
    return selected.reshape((-1, selected.shape[-1]))


def standardize_train_test(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def fit_ridge_probe(
    z_ego: np.ndarray,
    z_partner: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ridge_lambda: float,
):
    x_train = flatten_episodes(z_ego, train_idx)
    x_test = flatten_episodes(z_ego, test_idx)
    y_train = flatten_episodes(z_partner, train_idx)
    y_test = flatten_episodes(z_partner, test_idx)

    x_train, x_test = standardize_train_test(x_train, x_test)
    y_train, y_test = standardize_train_test(y_train, y_test)

    xtx = x_train.T @ x_train
    reg = ridge_lambda * np.eye(xtx.shape[0], dtype=x_train.dtype)
    weights = np.linalg.solve(xtx + reg, x_train.T @ y_train)
    pred = x_test @ weights

    error = pred - y_test
    sse = float(np.square(error).sum())
    sst = float(np.square(y_test).sum())
    mse = float(np.square(error).mean())
    baseline_mse = float(np.square(y_test).mean())
    normalized_mse = sse / sst if sst > 0.0 else np.nan
    r2 = 1.0 - normalized_mse if sst > 0.0 else np.nan

    return {
        "mse": mse,
        "baseline_mse": baseline_mse,
        "normalized_mse": normalized_mse,
        "r2": r2,
    }


def shuffled_partner_targets(z_partner: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.array(z_partner, copy=True)
    flat = shuffled.reshape((-1, shuffled.shape[-1]))
    flat = flat[rng.permutation(flat.shape[0])]
    return flat.reshape(shuffled.shape)


def mean_and_se(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    if values.size <= 1:
        return mean, np.nan
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean, se


def summarize_alignment_results(results: Dict) -> Dict:
    r2_mean, r2_se = mean_and_se(results["alignment_r2_matrix"].ravel())
    nmse_mean, nmse_se = mean_and_se(results["alignment_normalized_mse_matrix"].ravel())
    mse_mean, mse_se = mean_and_se(results["alignment_mse_matrix"].ravel())
    shuffled_r2_mean, shuffled_r2_se = mean_and_se(results["alignment_shuffled_r2_matrix"].ravel())

    return {
        "average_alignment_r2": r2_mean,
        "standard_error_alignment_r2": r2_se,
        "average_alignment_normalized_mse": nmse_mean,
        "standard_error_alignment_normalized_mse": nmse_se,
        "average_alignment_mse": mse_mean,
        "standard_error_alignment_mse": mse_se,
        "average_alignment_shuffled_r2": shuffled_r2_mean,
        "standard_error_alignment_shuffled_r2": shuffled_r2_se,
        "num_ego_agents": int(results["alignment_r2_matrix"].shape[0]),
        "num_partner_agents": int(results["alignment_r2_matrix"].shape[1]),
        "num_alignment_pairs": int(np.isfinite(results["alignment_r2_matrix"]).sum()),
        "ego_hidden_dim": int(results["ego_hidden_dim"]),
        "partner_hidden_dim": int(results["partner_hidden_dim"]),
        "train_episodes": int(results["train_episodes"]),
        "test_episodes": int(results["test_episodes"]),
    }


def evaluate_partner_alignment(
    config: Dict,
    ego_config: Dict,
    partner_config: Dict,
    ego_method_module,
    partner_method_module,
    ego_pool: Dict,
    partner_pool: Dict,
    rng,
):
    ego_pool_params = ego_pool["params"]
    partner_pool_params = partner_pool["params"]
    num_ego_agents = jax.tree_util.tree_leaves(ego_pool_params)[0].shape[0]
    num_partner_agents = jax.tree_util.tree_leaves(partner_pool_params)[0].shape[0]
    collector = make_partner_alignment_collector(
        config,
        ego_config,
        partner_config,
        ego_method_module,
        partner_method_module,
    )

    num_episodes = get_alignment_num_episodes(config, default=128)
    num_steps = int(
        config.get(
            "ALIGN_EVAL_NUM_STEPS",
            config.get("EVAL_NUM_STEPS", config["ENV_KWARGS"].get("max_steps", 400)),
        )
    )
    train_fraction = float(config.get("ALIGN_TRAIN_FRACTION", 0.7))
    ridge_lambda = float(config.get("ALIGN_RIDGE_LAMBDA", 1e-3))
    train_idx, test_idx = episode_split(num_episodes, train_fraction)

    r2_matrix = np.zeros((num_ego_agents, num_partner_agents), dtype=np.float32)
    normalized_mse_matrix = np.zeros((num_ego_agents, num_partner_agents), dtype=np.float32)
    mse_matrix = np.zeros((num_ego_agents, num_partner_agents), dtype=np.float32)
    baseline_mse_matrix = np.zeros((num_ego_agents, num_partner_agents), dtype=np.float32)
    shuffled_r2_matrix = np.zeros((num_ego_agents, num_partner_agents), dtype=np.float32)
    ego_hidden_dim = None
    partner_hidden_dim = None

    shuffle_seed = int(config.get("SEED", 0) + config.get("ALIGN_SHUFFLE_SEED_OFFSET", 20000))
    shuffle_rng = np.random.default_rng(shuffle_seed)
    pair_batch_size = int(config.get("ALIGN_PAIR_BATCH_SIZE", 1))
    max_pair_batch_size = config.get("ALIGN_MAX_PAIR_BATCH_SIZE", None)
    if max_pair_batch_size is not None:
        max_pair_batch_size = int(max_pair_batch_size)
        if pair_batch_size > max_pair_batch_size:
            print(
                "Capping ALIGN_PAIR_BATCH_SIZE "
                f"from {pair_batch_size} to {max_pair_batch_size} "
                "to avoid materializing too many hidden trajectories at once."
            )
            pair_batch_size = max_pair_batch_size
    print(
        "Using alignment rollout scale: "
        f"ALIGN_PAIR_BATCH_SIZE={pair_batch_size}, "
        f"ALIGN_EVAL_NUM_EPISODES={num_episodes}, "
        f"ALIGN_EVAL_NUM_STEPS={num_steps}"
    )
    collect_pair_batch_jit = jax.jit(jax.vmap(collector, in_axes=(0, 0, 0)))

    for pair_batch, valid_count in batched_pairs(
        num_ego_agents,
        num_partner_agents,
        pair_batch_size,
    ):
        ego_indices = np.array([i for i, _ in pair_batch], dtype=np.int32)
        partner_indices = np.array([j for _, j in pair_batch], dtype=np.int32)
        params_i = get_pool_params_batch(ego_pool_params, ego_indices)
        params_j = get_pool_params_batch(partner_pool_params, partner_indices)
        rng, pair_rng = jax.random.split(rng)
        pair_rngs = jax.random.split(pair_rng, len(pair_batch))
        traj_batch = collect_pair_batch_jit(pair_rngs, params_i, params_j)

        for batch_idx, (i, j) in enumerate(pair_batch[:valid_count]):
            z_ego = np.asarray(traj_batch["z_ego"][batch_idx], dtype=np.float32)
            z_partner = np.asarray(traj_batch["z_partner"][batch_idx], dtype=np.float32)
            if ego_hidden_dim is None:
                ego_hidden_dim = int(z_ego.shape[-1])
            if partner_hidden_dim is None:
                partner_hidden_dim = int(z_partner.shape[-1])

            metrics = fit_ridge_probe(
                z_ego,
                z_partner,
                train_idx,
                test_idx,
                ridge_lambda,
            )
            shuffled_metrics = fit_ridge_probe(
                z_ego,
                shuffled_partner_targets(z_partner, shuffle_rng),
                train_idx,
                test_idx,
                ridge_lambda,
            )

            r2_matrix[i, j] = metrics["r2"]
            normalized_mse_matrix[i, j] = metrics["normalized_mse"]
            mse_matrix[i, j] = metrics["mse"]
            baseline_mse_matrix[i, j] = metrics["baseline_mse"]
            shuffled_r2_matrix[i, j] = shuffled_metrics["r2"]
            print(
                f"ALIGN ego[{i:02d}] partner[{j:02d}] "
                f"R2={metrics['r2']:.6f} "
                f"NMSE={metrics['normalized_mse']:.6f} "
                f"shuffled_R2={shuffled_metrics['r2']:.6f}"
            )

    results = {
        "alignment_r2_matrix": r2_matrix,
        "alignment_normalized_mse_matrix": normalized_mse_matrix,
        "alignment_mse_matrix": mse_matrix,
        "alignment_baseline_mse_matrix": baseline_mse_matrix,
        "alignment_shuffled_r2_matrix": shuffled_r2_matrix,
        "ego_checkpoint_names": list(ego_pool["names"]),
        "partner_checkpoint_names": list(partner_pool["names"]),
        "ego_training_method": ego_config.get("TRAINING_METHOD", "sp"),
        "partner_training_method": partner_config.get("TRAINING_METHOD", "sp"),
        "result_name": default_result_name(config),
        "ego_checkpoints_prefix": ego_config.get("CHECKPOINTS_PREFIX", "./fcp_pool"),
        "partner_checkpoints_prefix": partner_config.get("CHECKPOINTS_PREFIX", "./fcp_pool"),
        "layout": config["ENV_KWARGS"]["layout"],
        "ego_hidden_dim": int(ego_hidden_dim),
        "partner_hidden_dim": int(partner_hidden_dim),
        "train_episodes": int(train_idx.size),
        "test_episodes": int(test_idx.size),
        "alignment_eval_num_episodes": num_episodes,
        "alignment_eval_num_steps": num_steps,
        "ridge_lambda": ridge_lambda,
        "train_fraction": train_fraction,
    }
    results["alignment_summary"] = summarize_alignment_results(results)
    return results


def print_alignment_summary(results: Dict):
    summary = results["alignment_summary"]
    print(
        "Average ego-to-partner alignment R2: "
        f"{summary['average_alignment_r2']:.6f} "
        f"+- {summary['standard_error_alignment_r2']:.6f} SE "
        f"over {summary['num_alignment_pairs']} ego-partner pairs"
    )
    print(
        "Average alignment normalized MSE: "
        f"{summary['average_alignment_normalized_mse']:.6f} "
        f"+- {summary['standard_error_alignment_normalized_mse']:.6f} SE"
    )
    print(
        "Average shuffled-control alignment R2: "
        f"{summary['average_alignment_shuffled_r2']:.6f} "
        f"+- {summary['standard_error_alignment_shuffled_r2']:.6f} SE"
    )


def save_alignment_results(results: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    summary_values = {
        key: value
        for key, value in results["alignment_summary"].items()
        if key not in {"ego_hidden_dim", "partner_hidden_dim", "train_episodes", "test_episodes"}
    }
    np.savez(
        os.path.join(save_dir, "alignment_results.npz"),
        alignment_r2_matrix=results["alignment_r2_matrix"],
        alignment_normalized_mse_matrix=results["alignment_normalized_mse_matrix"],
        alignment_mse_matrix=results["alignment_mse_matrix"],
        alignment_baseline_mse_matrix=results["alignment_baseline_mse_matrix"],
        alignment_shuffled_r2_matrix=results["alignment_shuffled_r2_matrix"],
        ego_checkpoint_names=np.array(results["ego_checkpoint_names"], dtype=object),
        partner_checkpoint_names=np.array(results["partner_checkpoint_names"], dtype=object),
        ego_training_method=results["ego_training_method"],
        partner_training_method=results["partner_training_method"],
        result_name=results["result_name"],
        ego_checkpoints_prefix=results["ego_checkpoints_prefix"],
        partner_checkpoints_prefix=results["partner_checkpoints_prefix"],
        layout=results["layout"],
        ego_hidden_dim=results["ego_hidden_dim"],
        partner_hidden_dim=results["partner_hidden_dim"],
        train_episodes=results["train_episodes"],
        test_episodes=results["test_episodes"],
        alignment_eval_num_episodes=results["alignment_eval_num_episodes"],
        alignment_eval_num_steps=results["alignment_eval_num_steps"],
        ridge_lambda=results["ridge_lambda"],
        train_fraction=results["train_fraction"],
        **summary_values,
    )
    np.savetxt(
        os.path.join(save_dir, "alignment_matrix_r2.csv"),
        results["alignment_r2_matrix"],
        delimiter=",",
        fmt="%.8f",
    )
    np.savetxt(
        os.path.join(save_dir, "alignment_matrix_nmse.csv"),
        results["alignment_normalized_mse_matrix"],
        delimiter=",",
        fmt="%.8f",
    )
    np.savetxt(
        os.path.join(save_dir, "alignment_matrix_shuffled_r2.csv"),
        results["alignment_shuffled_r2_matrix"],
        delimiter=",",
        fmt="%.8f",
    )
    with open(os.path.join(save_dir, "alignment_summary.csv"), "w") as f:
        f.write("metric,value\n")
        for key, value in results["alignment_summary"].items():
            f.write(f"{key},{value}\n")
        f.write(f"alignment_eval_num_episodes,{results['alignment_eval_num_episodes']}\n")
        f.write(f"alignment_eval_num_steps,{results['alignment_eval_num_steps']}\n")
        f.write(f"ridge_lambda,{results['ridge_lambda']}\n")
        f.write(f"train_fraction,{results['train_fraction']}\n")
    print(f"Saved alignment results to {save_dir}")


def run_pair_episode_with_states(
    config: Dict,
    ego_config: Dict,
    partner_config: Dict,
    ego_method_module,
    partner_method_module,
    params_a,
    params_b,
    rng,
):
    """
    Roll out one visualisation episode for a fixed ordered pair of agents.
    """
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = 1

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    ego_network_cls = get_network_class(ego_config.get("TRAINING_METHOD", "sp"), ego_method_module)
    ego_network = ego_network_cls(
        env.action_space(env.agents[0]).n,
        config=ego_config,
    )
    partner_network_cls = get_network_class(
        partner_config.get("TRAINING_METHOD", "sp"),
        partner_method_module,
    )
    partner_network = partner_network_cls(
        env.action_space(env.agents[0]).n,
        config=partner_config,
    )

    num_eval_steps = eval_config.get(
        "EVAL_NUM_STEPS",
        eval_config["ENV_KWARGS"].get("max_steps", 400),
    )
    sample_actions = eval_config.get("EVAL_SAMPLE_ACTIONS", False)

    rng, reset_rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng)

    hstate_a = initialize_hstate(ego_config, ego_method_module, batch_size=1)
    hstate_b = initialize_hstate(partner_config, partner_method_module, batch_size=1)
    done_batch = jnp.zeros((1,), dtype=bool)
    ep_return = jnp.zeros((1,), dtype=jnp.float32)

    def apply_policy(network, policy_config, params, hstate, obs, done, rng, priv_z=None):
        ac_in = [obs[jnp.newaxis, ...], done[jnp.newaxis, ...]]
        if uses_priv_z(policy_config):
            ac_in.append(priv_z[jnp.newaxis, ...])
        ac_in = tuple(ac_in)
        out = network.apply(params, hstate, ac_in)
        hstate, pi = out[0], out[1]
        if sample_actions:
            action = pi.sample(seed=rng).squeeze()
        else:
            action = jnp.argmax(pi.logits, axis=-1).squeeze()
        return hstate, action

    def step_fn(carry, _):
        obs, env_state, hstate_a, hstate_b, done_batch, ep_return, rng = carry
        rng, rng_a, rng_b, step_rng = jax.random.split(rng, 4)
        old_hstate_a = hstate_a
        old_hstate_b = hstate_b

        hstate_a, action_a = apply_policy(
            ego_network,
            ego_config,
            params_a,
            hstate_a,
            obs[env.agents[0]][None, ...],
            done_batch,
            rng_a,
            priv_z=old_hstate_b,
        )
        hstate_b, action_b = apply_policy(
            partner_network,
            partner_config,
            params_b,
            hstate_b,
            obs[env.agents[1]][None, ...],
            done_batch,
            rng_b,
            priv_z=old_hstate_a,
        )

        env_act = {
            env.agents[0]: action_a,
            env.agents[1]: action_b,
        }
        next_obs, next_env_state, reward, done, _ = env.step(step_rng, env_state, env_act)

        team_reward = jnp.array(
            [0.5 * (reward[env.agents[0]] + reward[env.agents[1]])],
            dtype=jnp.float32,
        )
        ep_return = ep_return + team_reward

        transition = {
            "state": env_state.env_state,
            "reward": team_reward[0],
            "done": done["__all__"],
            "action_a": action_a,
            "action_b": action_b,
        }
        next_carry = (
            next_obs,
            next_env_state,
            hstate_a,
            hstate_b,
            jnp.array([done["__all__"]], dtype=bool),
            ep_return,
            rng,
        )
        return next_carry, transition

    init_carry = (obsv, env_state, hstate_a, hstate_b, done_batch, ep_return, rng)
    final_carry, traj = jax.lax.scan(step_fn, init_carry, None, length=num_eval_steps)

    return {
        "episode_return": float(final_carry[5].mean()),
        "state_seq": traj["state"],
        "reward_seq": np.array(traj["reward"]),
        "done_seq": np.array(traj["done"]),
        "action_seq_a": np.array(traj["action_a"]),
        "action_seq_b": np.array(traj["action_b"]),
    }


def save_episode_mp4(
    state_seq,
    save_path: str,
    agent_view_size=None,
    fps: int = 4,
    tile_size: int = 32,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    viz = OvercookedV2Visualizer(tile_size=tile_size)
    frame_seq = viz.render_sequence(state_seq, agent_view_size=agent_view_size)
    frame_seq = np.array(frame_seq).astype(np.uint8)

    with imageio.get_writer(save_path, fps=fps) as writer:
        for frame in frame_seq:
            writer.append_data(frame)

    print(f"Saved mp4 to {save_path}")


def ranked_pairs_by_return(xp_matrix: np.ndarray, count: int = 5):
    finite_mask = np.isfinite(xp_matrix)
    flat_indices = np.flatnonzero(finite_mask.ravel())
    if flat_indices.size == 0:
        return [], []

    flat_returns = xp_matrix.ravel()[flat_indices]
    ascending = flat_indices[np.argsort(flat_returns, kind="stable")]
    worst = [
        (*np.unravel_index(int(flat_idx), xp_matrix.shape), float(xp_matrix.ravel()[flat_idx]))
        for flat_idx in ascending[:count]
    ]
    best = [
        (*np.unravel_index(int(flat_idx), xp_matrix.shape), float(xp_matrix.ravel()[flat_idx]))
        for flat_idx in ascending[-count:][::-1]
    ]
    return best, worst


def save_return_ranked_videos(
    config: Dict,
    ego_config: Dict,
    partner_config: Dict,
    ego_method_module,
    partner_method_module,
    ego_pool: Dict,
    partner_pool: Dict,
    results: Dict,
    save_dir: str,
    count: int = 5,
    fps: int = 4,
    tile_size: int = 32,
):
    os.makedirs(save_dir, exist_ok=True)

    best_pairs, worst_pairs = ranked_pairs_by_return(results["xp_matrix"], count=count)
    base_seed = config.get("SEED", 0) + config.get("XP_VIDEO_SEED_OFFSET", 5000)

    def save_ranked_group(label: str, pairs, seed_offset: int):
        for rank, (i, j, matrix_return) in enumerate(pairs, start=1):
            params_i = get_pool_params_i(ego_pool["params"], i)
            params_j = get_pool_params_i(partner_pool["params"], j)
            rng = jax.random.PRNGKey(base_seed + seed_offset + rank)
            print(
                f"Saving {label} return video #{rank}: "
                f"ego {i}, partner {j}, matrix return {matrix_return:.3f}"
            )
            episode = run_pair_episode_with_states(
                config,
                ego_config,
                partner_config,
                ego_method_module,
                partner_method_module,
                params_i,
                params_j,
                rng,
            )
            save_episode_mp4(
                episode["state_seq"],
                save_path=os.path.join(
                    save_dir,
                    f"{label}_return_rank_{rank:02d}_ego_{i:02d}_partner_{j:02d}_matrix_{matrix_return:.3f}.mp4",
                ),
                fps=fps,
                tile_size=tile_size,
            )
            print(
                f"Video rollout return for ego {i}, partner {j}: "
                f"{episode['episode_return']:.3f}"
            )

    save_ranked_group("best", best_pairs, seed_offset=0)
    save_ranked_group("worst", worst_pairs, seed_offset=1000)


def summarize_xp_matrix(xp_matrix: np.ndarray) -> Dict:
    def mean_and_se(values: np.ndarray):
        if values.size == 0:
            return np.nan, np.nan
        mean = float(values.mean())
        if values.size <= 1:
            return mean, np.nan
        se = float(values.std(ddof=1) / np.sqrt(values.size))
        return mean, se

    all_returns = np.asarray(xp_matrix, dtype=np.float64).ravel()
    all_returns = all_returns[np.isfinite(all_returns)]
    average_return, standard_error_return = mean_and_se(all_returns)

    diag_returns = np.asarray(np.diag(xp_matrix), dtype=np.float64)
    diag_returns = diag_returns[np.isfinite(diag_returns)]
    average_diagonal, standard_error_diagonal = mean_and_se(diag_returns)

    if xp_matrix.shape[0] == xp_matrix.shape[1]:
        off_diagonal_mask = ~np.eye(xp_matrix.shape[0], dtype=bool)
        off_diagonal_returns = np.asarray(xp_matrix[off_diagonal_mask], dtype=np.float64)
        off_diagonal_returns = off_diagonal_returns[np.isfinite(off_diagonal_returns)]
    else:
        off_diagonal_returns = np.asarray([], dtype=np.float64)
    average_off_diagonal, standard_error_off_diagonal = mean_and_se(off_diagonal_returns)

    return {
        "average_return": average_return,
        "standard_error_return": standard_error_return,
        "average_diagonal": average_diagonal,
        "standard_error_diagonal": standard_error_diagonal,
        "average_off_diagonal": average_off_diagonal,
        "standard_error_off_diagonal": standard_error_off_diagonal,
        "num_ego_agents": int(xp_matrix.shape[0]),
        "num_partner_agents": int(xp_matrix.shape[1]),
        "num_pairs": int(all_returns.size),
        "num_diagonal_pairs": int(diag_returns.size),
        "num_off_diagonal_pairs": int(off_diagonal_returns.size),
    }


def print_summary(results: Dict):
    summary = results["summary"]
    print(
        "Average ego-vs-partner performance: "
        f"{summary['average_return']:.3f} "
        f"+- {summary['standard_error_return']:.3f} SE "
        f"over {summary['num_pairs']} ordered pairs"
    )
    print(
        "Average diagonal performance: "
        f"{summary['average_diagonal']:.3f} "
        f"+- {summary['standard_error_diagonal']:.3f} SE "
        f"over {summary['num_diagonal_pairs']} pairs"
    )
    print(
        "Average off-diagonal performance "
        f"(only for square pools): {summary['average_off_diagonal']:.3f} "
        f"+- {summary['standard_error_off_diagonal']:.3f} SE "
        f"over {summary['num_off_diagonal_pairs']} ordered pairs"
    )


def save_results(results: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, "xp_results.npz"),
        xp_matrix=results["xp_matrix"],
        ego_checkpoint_names=np.array(results["ego_checkpoint_names"], dtype=object),
        partner_checkpoint_names=np.array(results["partner_checkpoint_names"], dtype=object),
        ego_training_method=results["ego_training_method"],
        partner_training_method=results["partner_training_method"],
        result_name=results["result_name"],
        ego_checkpoints_prefix=results["ego_checkpoints_prefix"],
        partner_checkpoints_prefix=results["partner_checkpoints_prefix"],
        layout=results["layout"],
        average_return=results["summary"]["average_return"],
        standard_error_return=results["summary"]["standard_error_return"],
        average_diagonal=results["summary"]["average_diagonal"],
        standard_error_diagonal=results["summary"]["standard_error_diagonal"],
        average_off_diagonal=results["summary"]["average_off_diagonal"],
        standard_error_off_diagonal=results["summary"]["standard_error_off_diagonal"],
        num_ego_agents=results["summary"]["num_ego_agents"],
        num_partner_agents=results["summary"]["num_partner_agents"],
        num_pairs=results["summary"]["num_pairs"],
        num_diagonal_pairs=results["summary"]["num_diagonal_pairs"],
        num_off_diagonal_pairs=results["summary"]["num_off_diagonal_pairs"],
    )
    np.savetxt(
        os.path.join(save_dir, "xp_matrix.csv"),
        results["xp_matrix"],
        delimiter=",",
        fmt="%.6f",
    )
    with open(os.path.join(save_dir, "ego_checkpoint_names.txt"), "w") as f:
        for idx, name in enumerate(results["ego_checkpoint_names"]):
            f.write(f"{idx},{name}\n")
    with open(os.path.join(save_dir, "partner_checkpoint_names.txt"), "w") as f:
        for idx, name in enumerate(results["partner_checkpoint_names"]):
            f.write(f"{idx},{name}\n")
    with open(os.path.join(save_dir, "summary.csv"), "w") as f:
        f.write("metric,value\n")
        for key, value in results["summary"].items():
            f.write(f"{key},{value}\n")
    print(f"Saved XP results to {save_dir}")


def plot_xp_matrix(results: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    matrix = results["xp_matrix"]
    ego_names = results["ego_checkpoint_names"]
    partner_names = results["partner_checkpoint_names"]

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    plt.title(
        "XP Matrix: "
        f"{results['ego_training_method']} ego vs "
        f"{results['partner_training_method']} partner / {results['layout']}"
    )
    plt.colorbar()
    plt.xlabel("Partner checkpoint")
    plt.ylabel("Ego checkpoint")
    if len(partner_names) <= 30:
        plt.xticks(range(len(partner_names)), partner_names, rotation=90, fontsize=8)
    else:
        plt.xticks(range(matrix.shape[1]))
    if len(ego_names) <= 30:
        plt.yticks(range(len(ego_names)), ego_names, fontsize=8)
    else:
        plt.yticks(range(matrix.shape[0]))
    plt.tight_layout()
    path = os.path.join(save_dir, "xp_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved XP heatmap to {path}")


@hydra.main(version_base=None, config_path="config/oc_extended/sp_pool", config_name="cramped_room2")
def main(config):
    config = OmegaConf.to_container(config)
    config["TRAINING_METHOD"] = config.get("TRAINING_METHOD", "sp")
    config["EGO_TRAINING_METHOD"] = config.get("EGO_TRAINING_METHOD", config["TRAINING_METHOD"])
    config["PARTNER_TRAINING_METHOD"] = config.get(
        "PARTNER_TRAINING_METHOD",
        config.get("PARTNER_METHOD", config["EGO_TRAINING_METHOD"]),
    )

    ego_config = make_role_config(config, "ego")
    partner_config = make_role_config(config, "partner")

    ego_method_module = get_method_module(ego_config["TRAINING_METHOD"])
    partner_method_module = get_method_module(partner_config["TRAINING_METHOD"])
    ego_pool = load_agent_pool(ego_config, ego_method_module)
    partner_pool = load_agent_pool(partner_config, partner_method_module)

    rng = jax.random.PRNGKey(config.get("SEED", 0) + config.get("XP_EVAL_SEED_OFFSET", 10000))
    evaluator = make_xp_evaluator(
        config,
        ego_config,
        partner_config,
        ego_method_module,
        partner_method_module,
        ego_pool,
        partner_pool,
    )
    results = evaluator(rng)
    results["summary"] = summarize_xp_matrix(results["xp_matrix"])
    print_summary(results)

    save_root = config.get("XP_SAVE_DIR", "./xp_results")
    save_dir = os.path.join(
        save_root,
        default_result_name(config),
        config["ENV_KWARGS"]["layout"],
    )
    save_results(results, save_dir)
    if config.get("XP_PLOT", True):
        plot_xp_matrix(results, save_dir)

    if config.get("XP_EVAL_ALIGNMENT", False):
        align_rng = jax.random.PRNGKey(
            config.get("SEED", 0) + config.get("ALIGN_EVAL_SEED_OFFSET", 30000)
        )
        alignment_results = evaluate_partner_alignment(
            config,
            ego_config,
            partner_config,
            ego_method_module,
            partner_method_module,
            ego_pool,
            partner_pool,
            align_rng,
        )
        print_alignment_summary(alignment_results)
        save_alignment_results(alignment_results, save_dir)

    if config.get("XP_SAVE_VIDEOS", False):
        video_dir = os.path.join(save_dir, "vids")
        save_return_ranked_videos(
            config,
            ego_config,
            partner_config,
            ego_method_module,
            partner_method_module,
            ego_pool,
            partner_pool,
            results,
            save_dir=video_dir,
            count=config.get("XP_NUM_BEST_WORST_VIDEOS", 5),
            fps=config.get("XP_VIDEO_FPS", 4),
            tile_size=config.get("XP_VIDEO_TILE_SIZE", 32),
        )


if __name__ == "__main__":
    main()
