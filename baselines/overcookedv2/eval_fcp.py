import os
import re
from typing import Dict, List, Tuple

import flax
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from omegaconf import OmegaConf

from baselines.overcookedv2 import train_fcp, train_sp


CHECKPOINT_RE = re.compile(r"baseline_seed_(?P<seed>\d+)_step_(?P<step>\d+)\.msgpack$")


def checkpoint_sort_key(name: str) -> Tuple[int, int, str]:
    match = CHECKPOINT_RE.match(os.path.basename(name))
    if match is None:
        return (10**12, 10**12, name)
    return (int(match.group("seed")), int(match.group("step")), name)


def checkpoint_dir(config: Dict, prefix_key: str, default_prefix: str) -> str:
    return os.path.join(
        config.get(prefix_key, default_prefix),
        config["ENV_KWARGS"]["layout"],
    )


def discover_checkpoints(
    config: Dict,
    prefix_key: str,
    default_prefix: str,
    latest_per_seed_key: str,
    explicit_names_key: str,
) -> List[str]:
    names = config.get(explicit_names_key, None)
    if names is not None:
        return list(names)

    ckpt_dir = checkpoint_dir(config, prefix_key, default_prefix)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    names = sorted(
        [name for name in os.listdir(ckpt_dir) if CHECKPOINT_RE.match(name)],
        key=checkpoint_sort_key,
    )
    if not names:
        raise FileNotFoundError(f"No baseline_seed_*_step_*.msgpack files in {ckpt_dir}")

    if config.get(latest_per_seed_key, True):
        latest_by_seed = {}
        for name in names:
            match = CHECKPOINT_RE.match(name)
            seed = int(match.group("seed"))
            step = int(match.group("step"))
            if seed not in latest_by_seed or step > latest_by_seed[seed][0]:
                latest_by_seed[seed] = (step, name)
        names = [latest_by_seed[seed][1] for seed in sorted(latest_by_seed)]

    return names


def initialize_hstate(config: Dict, batch_size: int):
    return train_sp.ScannedRNN.initialize_carry(batch_size, config["GRU_HIDDEN_DIM"])


def make_network_and_dummy_params(config: Dict, module):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)
    network = module.ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

    rng = jax.random.PRNGKey(config.get("SEED", 0))
    rng, reset_rng, init_rng = jax.random.split(rng, 3)
    batch_size = config.get("FCP_EVAL_NUM_ENVS", config.get("EVAL_NUM_ENVS", 128))
    reset_rng = jax.random.split(reset_rng, batch_size)
    obsv_init, _ = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

    hstate = initialize_hstate(config, batch_size)
    init_x = (
        obsv_init[env.agents[0]][jnp.newaxis, ...],
        jnp.zeros((1, batch_size), dtype=bool),
    )
    dummy_params = network.init(init_rng, hstate, init_x)
    return network, dummy_params


def load_pool(
    config: Dict,
    module,
    prefix_key: str,
    default_prefix: str,
    latest_per_seed_key: str,
    explicit_names_key: str,
):
    names = discover_checkpoints(
        config,
        prefix_key=prefix_key,
        default_prefix=default_prefix,
        latest_per_seed_key=latest_per_seed_key,
        explicit_names_key=explicit_names_key,
    )
    _, dummy_params = make_network_and_dummy_params(config, module)
    ckpt_dir = checkpoint_dir(config, prefix_key, default_prefix)

    loaded_params = []
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(ckpt_dir, name)
        with open(path, "rb") as f:
            loaded_params.append(flax.serialization.from_bytes(dummy_params, f.read()))
        print(f"LOADED: {path}")

    stacked_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *loaded_params
    )
    return {"params": stacked_params, "names": names}


def get_pool_params_i(pool_params, i: int):
    return jax.tree_util.tree_map(lambda x: x[i], pool_params)


def make_fcp_evaluator(config: Dict, ego_pool: Dict, partner_pool: Dict):
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = eval_config.get(
        "FCP_EVAL_NUM_ENVS",
        eval_config.get("EVAL_NUM_ENVS", 128),
    )

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    ego_network = train_fcp.ActorCriticRNN(
        env.action_space(env.agents[0]).n,
        config=eval_config,
    )
    partner_network = train_sp.ActorCriticRNN(
        env.action_space(env.agents[1]).n,
        config=eval_config,
    )

    num_eval_envs = eval_config["NUM_ENVS"]
    num_eval_steps = eval_config.get(
        "FCP_EVAL_NUM_STEPS",
        eval_config.get("EVAL_NUM_STEPS", eval_config["ENV_KWARGS"].get("max_steps", 400)),
    )
    num_eval_episodes = eval_config.get(
        "FCP_EVAL_NUM_EPISODES",
        eval_config.get("EVAL_NUM_EPISODES", 100),
    )
    sample_actions = eval_config.get(
        "FCP_EVAL_SAMPLE_ACTIONS",
        eval_config.get("EVAL_SAMPLE_ACTIONS", False),
    )

    def apply_policy(network, params, hstate, obs, done_batch, rng):
        ac_in = (obs[jnp.newaxis, :], done_batch[jnp.newaxis, :])
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        if sample_actions:
            action = pi.sample(seed=rng).squeeze(0)
        else:
            action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
        return hstate, action

    def evaluate_pair(rng, ego_params, partner_params):
        def run_episode(rng):
            rng, reset_key = jax.random.split(rng)
            reset_rng = jax.random.split(reset_key, num_eval_envs)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

            ego_hstate = initialize_hstate(eval_config, num_eval_envs)
            partner_hstate = initialize_hstate(eval_config, num_eval_envs)
            done_batch = jnp.zeros((num_eval_envs,), dtype=bool)
            ep_return = jnp.zeros((num_eval_envs,), dtype=jnp.float32)

            def step_fn(carry, _):
                obs, env_state, ego_hstate, partner_hstate, done_batch, ep_return, rng = carry
                rng, rng_ego, rng_partner, rng_step = jax.random.split(rng, 4)

                ego_hstate, ego_action = apply_policy(
                    ego_network,
                    ego_params,
                    ego_hstate,
                    obs[env.agents[0]],
                    done_batch,
                    rng_ego,
                )
                partner_hstate, partner_action = apply_policy(
                    partner_network,
                    partner_params,
                    partner_hstate,
                    obs[env.agents[1]],
                    done_batch,
                    rng_partner,
                )

                env_act = {
                    env.agents[0]: ego_action,
                    env.agents[1]: partner_action,
                }
                step_rng = jax.random.split(rng_step, num_eval_envs)
                next_obs, next_env_state, reward, done, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_rng, env_state, env_act)

                team_reward = 0.5 * (reward[env.agents[0]] + reward[env.agents[1]])
                ep_return = ep_return + team_reward
                return (
                    next_obs,
                    next_env_state,
                    ego_hstate,
                    partner_hstate,
                    done["__all__"],
                    ep_return,
                    rng,
                ), None

            init_carry = (
                obsv,
                env_state,
                ego_hstate,
                partner_hstate,
                done_batch,
                ep_return,
                rng,
            )
            final_carry, _ = jax.lax.scan(
                step_fn,
                init_carry,
                None,
                length=num_eval_steps,
            )
            return final_carry[5].mean()

        rngs = jax.random.split(rng, num_eval_episodes)
        returns = jax.vmap(run_episode)(rngs)
        return returns.mean()

    evaluate_pair_jit = jax.jit(evaluate_pair)

    def evaluator(rng):
        num_egos = jax.tree_util.tree_leaves(ego_pool["params"])[0].shape[0]
        num_partners = jax.tree_util.tree_leaves(partner_pool["params"])[0].shape[0]
        return_matrix = np.zeros((num_egos, num_partners), dtype=np.float32)

        for i in range(num_egos):
            ego_params = get_pool_params_i(ego_pool["params"], i)
            for j in range(num_partners):
                partner_params = get_pool_params_i(partner_pool["params"], j)
                rng, eval_rng = jax.random.split(rng)
                ret = float(evaluate_pair_jit(eval_rng, ego_params, partner_params))
                return_matrix[i, j] = ret
                print(f"FCP[{i:02d}] x SP[{j:02d}] = {ret:.3f}")

        return {
            "return_matrix": return_matrix,
            "ego_checkpoint_names": list(ego_pool["names"]),
            "partner_checkpoint_names": list(partner_pool["names"]),
            "layout": config["ENV_KWARGS"]["layout"],
            "ego_checkpoints_prefix": config.get("FCP_CHECKPOINTS_PREFIX", "checkpoints/fcp"),
            "partner_checkpoints_prefix": config.get(
                "SP_PARTNER_CHECKPOINTS_PREFIX",
                "checkpoints/sp",
            ),
        }

    return evaluator


def summarize_results(results: Dict) -> Dict:
    values = np.asarray(results["return_matrix"], dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan
    per_ego_mean = values.mean(axis=1)
    per_partner_mean = values.mean(axis=0)
    return {
        "average_fcp_with_sp": mean,
        "standard_error_fcp_with_sp": se,
        "num_fcp_egos": int(values.shape[0]),
        "num_sp_partners": int(values.shape[1]),
        "best_pair_return": float(values.max()),
        "worst_pair_return": float(values.min()),
        "best_ego_mean": float(per_ego_mean.max()),
        "worst_ego_mean": float(per_ego_mean.min()),
        "best_partner_mean": float(per_partner_mean.max()),
        "worst_partner_mean": float(per_partner_mean.min()),
    }


def save_results(results: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, "fcp_eval_results.npz"),
        return_matrix=results["return_matrix"],
        ego_checkpoint_names=np.array(results["ego_checkpoint_names"], dtype=object),
        partner_checkpoint_names=np.array(results["partner_checkpoint_names"], dtype=object),
        layout=results["layout"],
        ego_checkpoints_prefix=results["ego_checkpoints_prefix"],
        partner_checkpoints_prefix=results["partner_checkpoints_prefix"],
        **results["summary"],
    )
    np.savetxt(
        os.path.join(save_dir, "fcp_vs_sp_matrix.csv"),
        results["return_matrix"],
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
    print(f"Saved FCP eval results to {save_dir}")


def print_summary(results: Dict):
    summary = results["summary"]
    print(
        "Average FCP ego with SP partner performance: "
        f"{summary['average_fcp_with_sp']:.3f} "
        f"+- {summary['standard_error_fcp_with_sp']:.3f} SE "
        f"over {summary['num_fcp_egos'] * summary['num_sp_partners']} pairs"
    )
    print(
        "Pair return range: "
        f"{summary['worst_pair_return']:.3f} to {summary['best_pair_return']:.3f}"
    )


@hydra.main(version_base=None, config_path="config/oc_extended/phase2", config_name="cramped_room2")
def main(config):
    config = OmegaConf.to_container(config)

    ego_pool = load_pool(
        config,
        module=train_fcp,
        prefix_key="FCP_CHECKPOINTS_PREFIX",
        default_prefix="checkpoints/fcp",
        latest_per_seed_key="FCP_LATEST_PER_SEED",
        explicit_names_key="FCP_EGO_CHECKPOINTS",
    )
    partner_pool = load_pool(
        config,
        module=train_sp,
        prefix_key="SP_PARTNER_CHECKPOINTS_PREFIX",
        default_prefix="checkpoints/sp",
        latest_per_seed_key="SP_PARTNER_LATEST_PER_SEED",
        explicit_names_key="SP_PARTNER_CHECKPOINTS",
    )

    rng = jax.random.PRNGKey(config.get("SEED", 0) + config.get("FCP_EVAL_SEED_OFFSET", 20000))
    evaluator = make_fcp_evaluator(config, ego_pool, partner_pool)
    results = evaluator(rng)
    results["summary"] = summarize_results(results)
    print_summary(results)

    save_root = config.get("FCP_EVAL_SAVE_DIR", "./fcp_eval_results")
    save_dir = os.path.join(save_root, config["ENV_KWARGS"]["layout"])
    save_results(results, save_dir)


if __name__ == "__main__":
    main()
