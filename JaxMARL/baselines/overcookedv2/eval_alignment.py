import os
from typing import Dict, Tuple

import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from omegaconf import OmegaConf

from baselines.overcookedv2.eval_xp import (
    default_result_name,
    get_method_module,
    get_network_class,
    get_pool_params_i,
    initialize_hstate,
    load_agent_pool,
    uses_priv_z,
)


def hidden_to_array(hidden):
    if isinstance(hidden, tuple):
        return jnp.concatenate([hidden_to_array(h) for h in hidden], axis=-1)
    return hidden


def make_alignment_collector(config: Dict, method_module):
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = eval_config.get("EVAL_NUM_ENVS", 128)

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    network_cls = get_network_class(eval_config.get("TRAINING_METHOD", "sp"), method_module)
    network = network_cls(
        env.action_space(env.agents[0]).n,
        config=eval_config,
    )

    num_eval_steps = eval_config.get(
        "EVAL_NUM_STEPS",
        eval_config["ENV_KWARGS"].get("max_steps", 400),
    )
    num_eval_envs = eval_config["NUM_ENVS"]
    num_eval_episodes = eval_config.get("EVAL_NUM_EPISODES", 100)
    sample_actions = eval_config.get("EVAL_SAMPLE_ACTIONS", False)

    def apply_policy(params, hstate, obs, done_batch, rng, priv_z=None):
        ac_in = [obs[jnp.newaxis, :], done_batch[jnp.newaxis, :]]
        if uses_priv_z(eval_config):
            ac_in.append(priv_z[jnp.newaxis, :])
        out = network.apply(params, hstate, tuple(ac_in))
        hstate, pi = out[0], out[1]
        if sample_actions:
            action = pi.sample(seed=rng).squeeze(0)
        else:
            action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
        return hstate, action

    def collect_one_pair(rng, params_a, params_b):
        def run_one_episode(rng):
            rng, reset_key = jax.random.split(rng)
            reset_rng = jax.random.split(reset_key, num_eval_envs)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

            hstate_a = initialize_hstate(eval_config, method_module, num_eval_envs)
            hstate_b = initialize_hstate(eval_config, method_module, num_eval_envs)
            done_batch = jnp.zeros((num_eval_envs,), dtype=bool)

            def step_fn(carry, _):
                obs, env_state, hstate_a, hstate_b, done_batch, rng = carry
                rng, rng_a, rng_b, rng_step = jax.random.split(rng, 4)
                old_hstate_a = hstate_a
                old_hstate_b = hstate_b

                hstate_a, action_a = apply_policy(
                    params_a,
                    hstate_a,
                    obs[env.agents[0]],
                    done_batch,
                    rng_a,
                    priv_z=old_hstate_b,
                )
                hstate_b, action_b = apply_policy(
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
                step_rng = jax.random.split(rng_step, num_eval_envs)
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

        rngs = jax.random.split(rng, num_eval_episodes)
        return jax.vmap(run_one_episode)(rngs)

    return jax.jit(collect_one_pair)


def episode_split(num_episodes: int, train_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    if num_episodes < 2:
        raise ValueError("ALIGNMENT requires EVAL_NUM_EPISODES >= 2 for episode split.")
    train_count = int(np.floor(num_episodes * train_fraction))
    train_count = min(max(train_count, 1), num_episodes - 1)
    train_idx = np.arange(train_count)
    test_idx = np.arange(train_count, num_episodes)
    return train_idx, test_idx


def flatten_episodes(z: np.ndarray, episode_idx: np.ndarray) -> np.ndarray:
    selected = z[episode_idx]
    return selected.reshape((-1, selected.shape[-1]))


def standardize_train_test(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (test - mean) / std, mean, std


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

    x_train, x_test, _, _ = standardize_train_test(x_train, x_test)
    y_train, y_test, _, _ = standardize_train_test(y_train, y_test)

    xtx = x_train.T @ x_train
    reg = ridge_lambda * np.eye(xtx.shape[0], dtype=x_train.dtype)
    weights = np.linalg.solve(xtx + reg, x_train.T @ y_train)
    pred = x_test @ weights

    error = pred - y_test
    sse = float(np.square(error).sum())
    sst = float(np.square(y_test).sum())
    mse = float(np.square(error).mean())
    baseline_mse = float(np.square(y_test).mean())
    nmse = sse / sst if sst > 0.0 else np.nan
    r2 = 1.0 - nmse if sst > 0.0 else np.nan

    return {
        "mse": mse,
        "baseline_mse": baseline_mse,
        "normalized_mse": nmse,
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


def summarize_results(results: Dict) -> Dict:
    r2_mean, r2_se = mean_and_se(results["r2_matrix"].ravel())
    nmse_mean, nmse_se = mean_and_se(results["normalized_mse_matrix"].ravel())
    mse_mean, mse_se = mean_and_se(results["mse_matrix"].ravel())
    shuffled_r2_mean, shuffled_r2_se = mean_and_se(results["shuffled_r2_matrix"].ravel())

    return {
        "average_r2": r2_mean,
        "standard_error_r2": r2_se,
        "average_normalized_mse": nmse_mean,
        "standard_error_normalized_mse": nmse_se,
        "average_mse": mse_mean,
        "standard_error_mse": mse_se,
        "average_shuffled_r2": shuffled_r2_mean,
        "standard_error_shuffled_r2": shuffled_r2_se,
        "num_agents": int(results["r2_matrix"].shape[0]),
        "num_pairs": int(np.isfinite(results["r2_matrix"]).sum()),
        "hidden_dim": int(results["hidden_dim"]),
        "train_episodes": int(results["train_episodes"]),
        "test_episodes": int(results["test_episodes"]),
    }


def print_summary(results: Dict):
    summary = results["summary"]
    print(
        "Average alignment R2: "
        f"{summary['average_r2']:.6f} "
        f"+- {summary['standard_error_r2']:.6f} SE "
        f"over {summary['num_pairs']} ordered pairs"
    )
    print(
        "Average normalized MSE: "
        f"{summary['average_normalized_mse']:.6f} "
        f"+- {summary['standard_error_normalized_mse']:.6f} SE"
    )
    print(
        "Average shuffled-control R2: "
        f"{summary['average_shuffled_r2']:.6f} "
        f"+- {summary['standard_error_shuffled_r2']:.6f} SE"
    )


def save_results(results: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    summary_values = {
        key: value
        for key, value in results["summary"].items()
        if key not in {"hidden_dim", "train_episodes", "test_episodes"}
    }
    np.savez(
        os.path.join(save_dir, "alignment_results.npz"),
        r2_matrix=results["r2_matrix"],
        normalized_mse_matrix=results["normalized_mse_matrix"],
        mse_matrix=results["mse_matrix"],
        baseline_mse_matrix=results["baseline_mse_matrix"],
        shuffled_r2_matrix=results["shuffled_r2_matrix"],
        checkpoint_names=np.array(results["checkpoint_names"], dtype=object),
        training_method=results["training_method"],
        result_name=results["result_name"],
        checkpoints_prefix=results["checkpoints_prefix"],
        layout=results["layout"],
        hidden_dim=results["hidden_dim"],
        train_episodes=results["train_episodes"],
        test_episodes=results["test_episodes"],
        ridge_lambda=results["ridge_lambda"],
        train_fraction=results["train_fraction"],
        **summary_values,
    )
    np.savetxt(
        os.path.join(save_dir, "alignment_matrix_r2.csv"),
        results["r2_matrix"],
        delimiter=",",
        fmt="%.8f",
    )
    np.savetxt(
        os.path.join(save_dir, "alignment_matrix_nmse.csv"),
        results["normalized_mse_matrix"],
        delimiter=",",
        fmt="%.8f",
    )
    np.savetxt(
        os.path.join(save_dir, "alignment_matrix_shuffled_r2.csv"),
        results["shuffled_r2_matrix"],
        delimiter=",",
        fmt="%.8f",
    )
    with open(os.path.join(save_dir, "checkpoint_names.txt"), "w") as f:
        for idx, name in enumerate(results["checkpoint_names"]):
            f.write(f"{idx},{name}\n")
    with open(os.path.join(save_dir, "alignment_summary.csv"), "w") as f:
        f.write("metric,value\n")
        for key, value in results["summary"].items():
            f.write(f"{key},{value}\n")
        f.write(f"ridge_lambda,{results['ridge_lambda']}\n")
        f.write(f"train_fraction,{results['train_fraction']}\n")
    print(f"Saved alignment results to {save_dir}")


def evaluate_alignment(config: Dict, method_module, pool: Dict, rng):
    pool_params = pool["params"]
    num_agents = jax.tree_util.tree_leaves(pool_params)[0].shape[0]
    collector = make_alignment_collector(config, method_module)

    num_episodes = int(config.get("EVAL_NUM_EPISODES", 100))
    train_fraction = float(config.get("ALIGN_TRAIN_FRACTION", 0.7))
    ridge_lambda = float(config.get("ALIGN_RIDGE_LAMBDA", 1e-3))
    train_idx, test_idx = episode_split(num_episodes, train_fraction)

    r2_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
    normalized_mse_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
    mse_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
    baseline_mse_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
    shuffled_r2_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
    hidden_dim = None

    shuffle_seed = int(config.get("SEED", 0) + config.get("ALIGN_SHUFFLE_SEED_OFFSET", 20000))
    shuffle_rng = np.random.default_rng(shuffle_seed)

    for i in range(num_agents):
        params_i = get_pool_params_i(pool_params, i)
        for j in range(num_agents):
            params_j = get_pool_params_i(pool_params, j)
            rng, pair_rng = jax.random.split(rng)
            traj = collector(pair_rng, params_i, params_j)
            z_ego = np.asarray(traj["z_ego"], dtype=np.float32)
            z_partner = np.asarray(traj["z_partner"], dtype=np.float32)
            if hidden_dim is None:
                hidden_dim = int(z_ego.shape[-1])
            if z_ego.shape[-1] != z_partner.shape[-1]:
                raise ValueError(
                    "Ego and partner hidden dimensions must match for this probe: "
                    f"{z_ego.shape[-1]} vs {z_partner.shape[-1]}"
                )

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
                f"ALIGN[{i:02d}, {j:02d}] "
                f"R2={metrics['r2']:.6f} "
                f"NMSE={metrics['normalized_mse']:.6f} "
                f"shuffled_R2={shuffled_metrics['r2']:.6f}"
            )

    results = {
        "r2_matrix": r2_matrix,
        "normalized_mse_matrix": normalized_mse_matrix,
        "mse_matrix": mse_matrix,
        "baseline_mse_matrix": baseline_mse_matrix,
        "shuffled_r2_matrix": shuffled_r2_matrix,
        "checkpoint_names": list(pool["names"]),
        "training_method": config.get("TRAINING_METHOD", "sp"),
        "result_name": default_result_name(config),
        "checkpoints_prefix": config.get("CHECKPOINTS_PREFIX", "./fcp_pool"),
        "layout": config["ENV_KWARGS"]["layout"],
        "hidden_dim": int(hidden_dim),
        "train_episodes": int(train_idx.size),
        "test_episodes": int(test_idx.size),
        "ridge_lambda": ridge_lambda,
        "train_fraction": train_fraction,
    }
    results["summary"] = summarize_results(results)
    return results


@hydra.main(version_base=None, config_path="config/oc_extended/sp_pool_eval", config_name="cramped_room2")
def main(config):
    config = OmegaConf.to_container(config)
    config["TRAINING_METHOD"] = config.get("TRAINING_METHOD", "sp")

    method_module = get_method_module(config["TRAINING_METHOD"])
    pool = load_agent_pool(config, method_module)

    rng = jax.random.PRNGKey(
        config.get("SEED", 0) + config.get("ALIGN_EVAL_SEED_OFFSET", 10000)
    )
    results = evaluate_alignment(config, method_module, pool, rng)
    print_summary(results)

    save_root = config.get("ALIGN_SAVE_DIR", "./alignment_results")
    save_dir = os.path.join(
        save_root,
        default_result_name(config),
        config["ENV_KWARGS"]["layout"],
    )
    save_results(results, save_dir)


if __name__ == "__main__":
    main()
