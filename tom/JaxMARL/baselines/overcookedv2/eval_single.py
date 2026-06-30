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
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from omegaconf import OmegaConf


CHECKPOINT_RE = re.compile(r"baseline_seed_(?P<seed>\d+)_step_(?P<step>\d+)\.msgpack$")


def get_method_module(method: str):
    method_key = method.lower()
    if method_key in {"single", "train_single"}:
        try:
            return importlib.import_module("baselines.overcookedv2.train_single")
        except ModuleNotFoundError:
            return importlib.import_module("baselines.overcookedv2.train_ph1_single")
    if method_key in {"ph1_single", "ppo_single"}:
        return importlib.import_module("baselines.overcookedv2.train_ph1_single")
    raise ValueError(
        f"Unknown SINGLE_TRAINING_METHOD={method!r}. "
        "Use one of: single, train_single, ph1_single, ppo_single."
    )


def checkpoint_dir(config: Dict) -> str:
    layout_name = config["ENV_KWARGS"]["layout"]
    checkpoints_prefix = config.get("CHECKPOINTS_PREFIX", "./checkpoints/single")
    return os.path.join(checkpoints_prefix, layout_name)


def default_result_name(config: Dict) -> str:
    result_name = config.get("SINGLE_RESULT_NAME", None)
    if result_name is not None:
        return str(result_name)
    return str(config.get("SINGLE_TRAINING_METHOD", "single"))


def checkpoint_sort_key(name: str) -> Tuple[int, int, str]:
    match = CHECKPOINT_RE.match(name)
    if match is None:
        return (10**12, 10**12, name)
    return (int(match.group("seed")), int(match.group("step")), name)


def discover_checkpoints(config: Dict) -> List[str]:
    ckpt_dir = checkpoint_dir(config)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {ckpt_dir}. "
            "Set CHECKPOINTS_PREFIX to the single-agent training output directory."
        )

    names = sorted(
        [name for name in os.listdir(ckpt_dir) if CHECKPOINT_RE.match(name)],
        key=checkpoint_sort_key,
    )
    if not names:
        raise FileNotFoundError(f"No baseline_seed_*_step_*.msgpack files found in {ckpt_dir}")

    if config.get("SINGLE_LATEST_PER_SEED", True):
        latest_by_seed = {}
        for name in names:
            match = CHECKPOINT_RE.match(name)
            seed = int(match.group("seed"))
            step = int(match.group("step"))
            if seed not in latest_by_seed or step > latest_by_seed[seed][0]:
                latest_by_seed[seed] = (step, name)
        names = [latest_by_seed[seed][1] for seed in sorted(latest_by_seed)]

    seed_filter = config.get("SINGLE_SEEDS", None)
    if seed_filter is not None:
        seed_filter = {int(seed) for seed in seed_filter}
        names = [
            name
            for name in names
            if int(CHECKPOINT_RE.match(name).group("seed")) in seed_filter
        ]

    if not names:
        raise ValueError("No checkpoints remain after applying SINGLE_SEEDS.")
    return names


def initialize_hstate(config: Dict, method_module, batch_size: int):
    return method_module.ScannedRNN.initialize_carry(
        batch_size,
        config["GRU_HIDDEN_DIM"],
    )


def make_network_and_dummy_params(config: Dict, method_module):
    eval_config = dict(config)
    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    network = method_module.ActorCriticRNN(
        env.action_space(env.agents[0]).n,
        config=eval_config,
    )

    rng = jax.random.PRNGKey(eval_config.get("SEED", 0))
    rng, reset_rng, init_rng = jax.random.split(rng, 3)
    batch_size = eval_config.get("SINGLE_EVAL_NUM_ENVS", eval_config.get("EVAL_NUM_ENVS", 1))
    reset_rng = jax.random.split(reset_rng, batch_size)
    obsv_init, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

    hstate = initialize_hstate(eval_config, method_module, batch_size)
    init_x = (
        obsv_init[env.agents[0]][jnp.newaxis, ...],
        jnp.zeros((1, batch_size), dtype=bool),
    )
    dummy_params = network.init(init_rng, hstate, init_x)
    return network, dummy_params


def load_agent_pool(config: Dict, method_module):
    names = config.get("SINGLE_CHECKPOINTS", None)
    if names is None:
        names = discover_checkpoints(config)
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


def make_single_evaluator(config: Dict, method_module, pool: Dict):
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = eval_config.get(
        "SINGLE_EVAL_NUM_ENVS",
        eval_config.get("EVAL_NUM_ENVS", 128),
    )

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    network = method_module.ActorCriticRNN(
        env.action_space(env.agents[0]).n,
        config=eval_config,
    )

    pool_params = pool["params"]
    num_agents = jax.tree_util.tree_leaves(pool_params)[0].shape[0]
    num_eval_steps = eval_config.get(
        "SINGLE_EVAL_NUM_STEPS",
        eval_config.get("EVAL_NUM_STEPS", eval_config["ENV_KWARGS"].get("max_steps", 400)),
    )
    num_eval_envs = eval_config["NUM_ENVS"]
    num_eval_episodes = eval_config.get(
        "SINGLE_EVAL_NUM_EPISODES",
        eval_config.get("EVAL_NUM_EPISODES", 100),
    )
    sample_actions = eval_config.get(
        "SINGLE_EVAL_SAMPLE_ACTIONS",
        eval_config.get("EVAL_SAMPLE_ACTIONS", False),
    )

    def apply_policy(params, hstate, obs, done_batch, rng):
        ac_in = (obs[jnp.newaxis, :], done_batch[jnp.newaxis, :])
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        if sample_actions:
            action = pi.sample(seed=rng).squeeze(0)
        else:
            action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
        return hstate, action

    def evaluate_one_agent(rng, params):
        def run_one_episode(rng):
            rng, reset_key = jax.random.split(rng)
            reset_rng = jax.random.split(reset_key, num_eval_envs)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

            hstate = initialize_hstate(eval_config, method_module, num_eval_envs)
            done_batch = jnp.zeros((num_eval_envs,), dtype=bool)
            ep_return = jnp.zeros((num_eval_envs,), dtype=jnp.float32)

            def step_fn(carry, _):
                obs, env_state, hstate, done_batch, ep_return, rng = carry
                rng, rng_action, rng_step = jax.random.split(rng, 3)

                hstate, action = apply_policy(
                    params,
                    hstate,
                    obs[env.agents[0]],
                    done_batch,
                    rng_action,
                )
                env_act = {env.agents[0]: action}
                step_rng = jax.random.split(rng_step, num_eval_envs)
                next_obs, next_env_state, reward, done, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_rng, env_state, env_act)

                ep_return = ep_return + reward[env.agents[0]]
                return (
                    next_obs,
                    next_env_state,
                    hstate,
                    done["__all__"],
                    ep_return,
                    rng,
                ), None

            init_carry = (obsv, env_state, hstate, done_batch, ep_return, rng)
            final_carry, _ = jax.lax.scan(step_fn, init_carry, None, length=num_eval_steps)
            return final_carry[4].mean()

        rngs = jax.random.split(rng, num_eval_episodes)
        returns = jax.vmap(run_one_episode)(rngs)
        return returns.mean()

    evaluate_one_agent_jit = jax.jit(evaluate_one_agent)

    def evaluator(rng):
        returns = np.zeros((num_agents,), dtype=np.float32)
        for i in range(num_agents):
            params_i = get_pool_params_i(pool_params, i)
            rng, eval_rng = jax.random.split(rng)
            ret_i = float(evaluate_one_agent_jit(eval_rng, params_i))
            returns[i] = ret_i
            print(f"Return[{i:02d}] = {ret_i:.3f}")

        return {
            "returns": returns,
            "checkpoint_names": list(pool["names"]),
            "training_method": config.get("SINGLE_TRAINING_METHOD", "single"),
            "result_name": default_result_name(config),
            "checkpoints_prefix": config.get("CHECKPOINTS_PREFIX", "./checkpoints/single"),
            "layout": config["ENV_KWARGS"]["layout"],
        }

    return evaluator


def run_single_episode_with_states(config: Dict, method_module, params, rng):
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = 1

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    network = method_module.ActorCriticRNN(
        env.action_space(env.agents[0]).n,
        config=eval_config,
    )

    num_eval_steps = eval_config.get(
        "SINGLE_EVAL_NUM_STEPS",
        eval_config.get("EVAL_NUM_STEPS", eval_config["ENV_KWARGS"].get("max_steps", 400)),
    )
    sample_actions = eval_config.get(
        "SINGLE_EVAL_SAMPLE_ACTIONS",
        eval_config.get("EVAL_SAMPLE_ACTIONS", False),
    )

    rng, reset_rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng)

    hstate = initialize_hstate(eval_config, method_module, batch_size=1)
    done_batch = jnp.zeros((1,), dtype=bool)
    ep_return = jnp.zeros((1,), dtype=jnp.float32)

    def apply_policy(params, hstate, obs, done, rng):
        ac_in = (obs[jnp.newaxis, ...], done[jnp.newaxis, ...])
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        if sample_actions:
            action = pi.sample(seed=rng).squeeze()
        else:
            action = jnp.argmax(pi.logits, axis=-1).squeeze()
        return hstate, action

    def step_fn(carry, _):
        obs, env_state, hstate, done_batch, ep_return, rng = carry
        rng, rng_action, step_rng = jax.random.split(rng, 3)

        hstate, action = apply_policy(
            params,
            hstate,
            obs[env.agents[0]][None, ...],
            done_batch,
            rng_action,
        )

        env_act = {env.agents[0]: action}
        next_obs, next_env_state, reward, done, _ = env.step(step_rng, env_state, env_act)

        step_reward = jnp.array([reward[env.agents[0]]], dtype=jnp.float32)
        ep_return = ep_return + step_reward

        transition = {
            "state": env_state.env_state,
            "reward": step_reward[0],
            "done": done["__all__"],
            "action": action,
        }
        next_carry = (
            next_obs,
            next_env_state,
            hstate,
            jnp.array([done["__all__"]], dtype=bool),
            ep_return,
            rng,
        )
        return next_carry, transition

    init_carry = (obsv, env_state, hstate, done_batch, ep_return, rng)
    final_carry, traj = jax.lax.scan(step_fn, init_carry, None, length=num_eval_steps)

    return {
        "episode_return": float(final_carry[4].mean()),
        "state_seq": traj["state"],
        "reward_seq": np.array(traj["reward"]),
        "done_seq": np.array(traj["done"]),
        "action_seq": np.array(traj["action"]),
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


def ranked_agents_by_return(returns: np.ndarray, count: int = 5):
    finite_indices = np.flatnonzero(np.isfinite(returns))
    if finite_indices.size == 0:
        return [], []

    ascending = finite_indices[np.argsort(returns[finite_indices], kind="stable")]
    worst = [(int(i), float(returns[i])) for i in ascending[:count]]
    best = [(int(i), float(returns[i])) for i in ascending[-count:][::-1]]
    return best, worst


def save_return_ranked_videos(
    config: Dict,
    method_module,
    pool: Dict,
    results: Dict,
    save_dir: str,
    count: int = 5,
    fps: int = 4,
    tile_size: int = 32,
):
    os.makedirs(save_dir, exist_ok=True)

    best_agents, worst_agents = ranked_agents_by_return(results["returns"], count=count)
    base_seed = config.get("SEED", 0) + config.get("SINGLE_VIDEO_SEED_OFFSET", 5000)

    def save_ranked_group(label: str, agents, seed_offset: int):
        for rank, (i, mean_return) in enumerate(agents, start=1):
            params_i = get_pool_params_i(pool["params"], i)
            rng = jax.random.PRNGKey(base_seed + seed_offset + rank)
            print(
                f"Saving {label} return video #{rank}: "
                f"agent {i}, mean return {mean_return:.3f}"
            )
            episode = run_single_episode_with_states(config, method_module, params_i, rng)
            save_episode_mp4(
                episode["state_seq"],
                save_path=os.path.join(
                    save_dir,
                    f"{label}_return_rank_{rank:02d}_agent_{i:02d}_mean_{mean_return:.3f}.mp4",
                ),
                fps=fps,
                tile_size=tile_size,
            )
            print(f"Video rollout return for agent {i}: {episode['episode_return']:.3f}")

    save_ranked_group("best", best_agents, seed_offset=0)
    save_ranked_group("worst", worst_agents, seed_offset=1000)


def summarize_returns(returns: np.ndarray) -> Dict:
    finite_returns = np.asarray(returns, dtype=np.float64)
    finite_returns = finite_returns[np.isfinite(finite_returns)]
    if finite_returns.size == 0:
        return {
            "average_return": np.nan,
            "standard_error_return": np.nan,
            "num_agents": 0,
        }

    average_return = float(finite_returns.mean())
    standard_error_return = (
        float(finite_returns.std(ddof=1) / np.sqrt(finite_returns.size))
        if finite_returns.size > 1
        else np.nan
    )
    return {
        "average_return": average_return,
        "standard_error_return": standard_error_return,
        "num_agents": int(finite_returns.size),
    }


def print_summary(results: Dict):
    summary = results["summary"]
    print(
        "Average single-agent performance: "
        f"{summary['average_return']:.3f} "
        f"+- {summary['standard_error_return']:.3f} SE "
        f"over {summary['num_agents']} checkpoints"
    )


def save_results(results: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, "single_results.npz"),
        returns=results["returns"],
        checkpoint_names=np.array(results["checkpoint_names"], dtype=object),
        training_method=results["training_method"],
        result_name=results["result_name"],
        checkpoints_prefix=results["checkpoints_prefix"],
        layout=results["layout"],
        average_return=results["summary"]["average_return"],
        standard_error_return=results["summary"]["standard_error_return"],
        num_agents=results["summary"]["num_agents"],
    )
    np.savetxt(
        os.path.join(save_dir, "single_returns.csv"),
        results["returns"],
        delimiter=",",
        fmt="%.6f",
    )
    with open(os.path.join(save_dir, "checkpoint_names.txt"), "w") as f:
        for idx, name in enumerate(results["checkpoint_names"]):
            f.write(f"{idx},{name}\n")
    with open(os.path.join(save_dir, "summary.csv"), "w") as f:
        f.write("metric,value\n")
        for key, value in results["summary"].items():
            f.write(f"{key},{value}\n")
    print(f"Saved single-agent results to {save_dir}")


def plot_returns(results: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    returns = results["returns"]

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(returns)), returns)
    plt.title(f"Single-Agent Returns: {results['training_method']} / {results['layout']}")
    plt.xlabel("Checkpoint index")
    plt.ylabel("Return")
    plt.tight_layout()
    path = os.path.join(save_dir, "single_returns.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved single-agent return plot to {path}")


@hydra.main(version_base=None, config_path="config/oc_single/train", config_name="cramped_room2")
def main(config):
    config = OmegaConf.to_container(config)
    config["SINGLE_TRAINING_METHOD"] = config.get("SINGLE_TRAINING_METHOD", "single")

    method_module = get_method_module(config["SINGLE_TRAINING_METHOD"])
    pool = load_agent_pool(config, method_module)

    rng = jax.random.PRNGKey(
        config.get("SEED", 0) + config.get("SINGLE_EVAL_SEED_OFFSET", 10000)
    )
    evaluator = make_single_evaluator(config, method_module, pool)
    results = evaluator(rng)
    results["summary"] = summarize_returns(results["returns"])
    print_summary(results)

    save_root = config.get("SINGLE_SAVE_DIR", "./single_results")
    save_dir = os.path.join(
        save_root,
        default_result_name(config),
        config["ENV_KWARGS"]["layout"],
    )
    save_results(results, save_dir)
    if config.get("SINGLE_PLOT", True):
        plot_returns(results, save_dir)
    if config.get("SINGLE_SAVE_VIDEOS", False):
        video_dir = os.path.join(save_dir, "vids")
        save_return_ranked_videos(
            config,
            method_module,
            pool,
            results,
            save_dir=video_dir,
            count=config.get("SINGLE_NUM_BEST_WORST_VIDEOS", 5),
            fps=config.get("SINGLE_VIDEO_FPS", 4),
            tile_size=config.get("SINGLE_VIDEO_TILE_SIZE", 32),
        )


if __name__ == "__main__":
    main()
