import os
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper


def get_pool_params_batch(pool_params, indices: np.ndarray):
    return jax.tree_util.tree_map(lambda x: x[indices], pool_params)


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
    get_network_class_fn: Callable,
    initialize_hstate_fn: Callable,
    uses_priv_z_fn: Callable,
):
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = get_alignment_num_episodes(eval_config, default=128)

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    ego_network_cls = get_network_class_fn(
        ego_config.get("TRAINING_METHOD", "sp"),
        ego_method_module,
    )
    ego_network = ego_network_cls(
        env.action_space(env.agents[0]).n,
        config=ego_config,
    )
    partner_network_cls = get_network_class_fn(
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
        if uses_priv_z_fn(policy_config):
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

        hstate_a = initialize_hstate_fn(ego_config, ego_method_module, num_eval_episodes)
        hstate_b = initialize_hstate_fn(partner_config, partner_method_module, num_eval_episodes)
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

    summary = {
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
    if "hidden_dim" in results:
        summary["hidden_dim"] = int(results["hidden_dim"])
        summary["num_agents"] = int(results["alignment_r2_matrix"].shape[0])
    return summary


def evaluate_partner_alignment(
    config: Dict,
    ego_config: Dict,
    partner_config: Dict,
    ego_method_module,
    partner_method_module,
    ego_pool: Dict,
    partner_pool: Dict,
    rng,
    get_network_class_fn: Callable,
    initialize_hstate_fn: Callable,
    uses_priv_z_fn: Callable,
    default_result_name_fn: Callable,
    zsc_compatible: bool = False,
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
        get_network_class_fn,
        initialize_hstate_fn,
        uses_priv_z_fn,
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
        "result_name": default_result_name_fn(config),
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
    if zsc_compatible:
        results["checkpoint_names"] = list(ego_pool["names"])
        results["training_method"] = ego_config.get("TRAINING_METHOD", "sp")
        results["checkpoints_prefix"] = ego_config.get("CHECKPOINTS_PREFIX", "./fcp_pool")
        if ego_hidden_dim == partner_hidden_dim:
            results["hidden_dim"] = int(ego_hidden_dim)
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
        if key
        not in {
            "ego_hidden_dim",
            "partner_hidden_dim",
            "hidden_dim",
            "train_episodes",
            "test_episodes",
        }
    }
    save_values = {
        "alignment_r2_matrix": results["alignment_r2_matrix"],
        "alignment_normalized_mse_matrix": results["alignment_normalized_mse_matrix"],
        "alignment_mse_matrix": results["alignment_mse_matrix"],
        "alignment_baseline_mse_matrix": results["alignment_baseline_mse_matrix"],
        "alignment_shuffled_r2_matrix": results["alignment_shuffled_r2_matrix"],
        "ego_checkpoint_names": np.array(results["ego_checkpoint_names"], dtype=object),
        "partner_checkpoint_names": np.array(results["partner_checkpoint_names"], dtype=object),
        "ego_training_method": results["ego_training_method"],
        "partner_training_method": results["partner_training_method"],
        "result_name": results["result_name"],
        "ego_checkpoints_prefix": results["ego_checkpoints_prefix"],
        "partner_checkpoints_prefix": results["partner_checkpoints_prefix"],
        "layout": results["layout"],
        "ego_hidden_dim": results["ego_hidden_dim"],
        "partner_hidden_dim": results["partner_hidden_dim"],
        "train_episodes": results["train_episodes"],
        "test_episodes": results["test_episodes"],
        "alignment_eval_num_episodes": results["alignment_eval_num_episodes"],
        "alignment_eval_num_steps": results["alignment_eval_num_steps"],
        "ridge_lambda": results["ridge_lambda"],
        "train_fraction": results["train_fraction"],
        **summary_values,
    }
    if "checkpoint_names" in results:
        save_values["checkpoint_names"] = np.array(results["checkpoint_names"], dtype=object)
        save_values["training_method"] = results["training_method"]
        save_values["checkpoints_prefix"] = results["checkpoints_prefix"]
    if "hidden_dim" in results:
        save_values["hidden_dim"] = results["hidden_dim"]

    np.savez(os.path.join(save_dir, "alignment_results.npz"), **save_values)
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
    with open(os.path.join(save_dir, "ego_checkpoint_names.txt"), "w") as f:
        for idx, name in enumerate(results["ego_checkpoint_names"]):
            f.write(f"{idx},{name}\n")
    with open(os.path.join(save_dir, "partner_checkpoint_names.txt"), "w") as f:
        for idx, name in enumerate(results["partner_checkpoint_names"]):
            f.write(f"{idx},{name}\n")
    if "checkpoint_names" in results:
        with open(os.path.join(save_dir, "checkpoint_names.txt"), "w") as f:
            for idx, name in enumerate(results["checkpoint_names"]):
                f.write(f"{idx},{name}\n")
    with open(os.path.join(save_dir, "alignment_summary.csv"), "w") as f:
        f.write("metric,value\n")
        for key, value in results["alignment_summary"].items():
            f.write(f"{key},{value}\n")
        f.write(f"alignment_eval_num_episodes,{results['alignment_eval_num_episodes']}\n")
        f.write(f"alignment_eval_num_steps,{results['alignment_eval_num_steps']}\n")
        f.write(f"ridge_lambda,{results['ridge_lambda']}\n")
        f.write(f"train_fraction,{results['train_fraction']}\n")
    print(f"Saved alignment results to {save_dir}")
