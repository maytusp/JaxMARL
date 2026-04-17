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
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from omegaconf import OmegaConf


import matplotlib.pyplot as plt
import imageio

from sklearn.cluster import SpectralClustering
from .networks import ActorCriticRNN, ScannedRNN

from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer


def load_partner_pool(config, checkpoint_names, dummy_params):
    layout_name = config["ENV_KWARGS"]["layout"]
    checkpoints_prefix = config.get("CHECKPOINTS_PREFIX", "./checkpoints/sp_pools")
    pool_dir = os.path.join(checkpoints_prefix, layout_name)

    loaded_params = []
    loaded_names = []

    for ckpt_name in checkpoint_names:
        ckpt_path = os.path.join(pool_dir, ckpt_name)
        with open(ckpt_path, "rb") as f:
            p = flax.serialization.from_bytes(dummy_params, f.read())
        loaded_params.append(p)
        loaded_names.append(ckpt_name)
        print(f"LOADED: {ckpt_name}")

    stacked_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *loaded_params
    )

    return {
        "params": stacked_params,
        "names": loaded_names,
    }



def make_zsc_evaluator(config, eval_pool):
    """
    Computes:
      1) XP matrix: raw pairwise cross-play returns
      2) Similarity matrix: normalized from XP matrix
      3) Optional spectral clustering on similarity matrix

    Assumptions:
      - all checkpoints in eval_pool are old SP checkpoints using ActorCriticRNN
      - 2-agent Overcooked setting
    """
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = config.get("EVAL_NUM_ENVS", 128)

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    network = ActorCriticRNN(
        env.action_space(env.agents[0]).n,
        config=eval_config,
        rnn_type="gru",
    )

    pool_params = eval_pool["params"]
    pool_names = eval_pool["names"]
    num_agents_in_pool = jax.tree_util.tree_leaves(pool_params)[0].shape[0]

    num_eval_steps = eval_config.get("EVAL_NUM_STEPS", eval_config["NUM_STEPS"])
    num_eval_envs = eval_config["NUM_ENVS"]
    num_eval_episodes = eval_config.get("EVAL_NUM_EPISODES", 100)

    def _get_params_i(params_tree, i):
        return jax.tree_util.tree_map(lambda x: x[i], params_tree)

    def evaluate_one_order(rng, params_a, params_b):
        """
        Evaluate agent A in slot 0 and agent B in slot 1.
        Return mean team reward over episodes.
        """
        def run_one_episode(rng):
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, num_eval_envs)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

            hstate_a = ScannedRNN.initialize_carry(
                eval_config, rnn_type="gru", batch_size=num_eval_envs
            )
            hstate_b = ScannedRNN.initialize_carry(
                eval_config, rnn_type="gru", batch_size=num_eval_envs
            )

            done_batch = jnp.zeros((num_eval_envs,), dtype=bool)
            ep_return = jnp.zeros((num_eval_envs,), dtype=jnp.float32)

            def _step_fn(carry, _):
                obs, env_state, hstate_a, hstate_b, done_batch, ep_return, rng = carry
                rng, rng_step = jax.random.split(rng, 2)

                obs_a = obs[env.agents[0]]
                obs_b = obs[env.agents[1]]

                ac_in_a = (obs_a[jnp.newaxis, :], done_batch[jnp.newaxis, :])
                ac_in_b = (obs_b[jnp.newaxis, :], done_batch[jnp.newaxis, :])

                hstate_a, pi_a, _ = network.apply(params_a, hstate_a, ac_in_a)
                hstate_b, pi_b, _ = network.apply(params_b, hstate_b, ac_in_b)

                # Greedy evaluation
                action_a = jnp.argmax(pi_a.logits, axis=-1).squeeze(0)
                action_b = jnp.argmax(pi_b.logits, axis=-1).squeeze(0)

                env_act = {
                    env.agents[0]: action_a,
                    env.agents[1]: action_b,
                }

                step_rng = jax.random.split(rng_step, num_eval_envs)
                next_obs, next_env_state, reward, done, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_rng, env_state, env_act)

                # Team reward
                reward_team = 0.5 * (
                    reward[env.agents[0]] + reward[env.agents[1]]
                )
                new_done_batch = done["__all__"]
                ep_return = ep_return + reward_team

                new_carry = (
                    next_obs,
                    next_env_state,
                    hstate_a,
                    hstate_b,
                    new_done_batch,
                    ep_return,
                    rng,
                )
                return new_carry, None

            init_carry = (
                obsv,
                env_state,
                hstate_a,
                hstate_b,
                done_batch,
                ep_return,
                rng,
            )

            final_carry, _ = jax.lax.scan(
                _step_fn, init_carry, None, length=num_eval_steps
            )
            ep_return = final_carry[5]
            return ep_return.mean()

        rngs = jax.random.split(rng, num_eval_episodes)
        returns = jax.vmap(run_one_episode)(rngs)
        return returns.mean()

    evaluate_one_order_jit = jax.jit(evaluate_one_order)

    def build_xp_matrix(rng):
        """
        XP[i, j] = J(pi_i in slot0, pi_j in slot1)
        This is directional and not necessarily symmetric.
        """
        xp_matrix = np.zeros((num_agents_in_pool, num_agents_in_pool), dtype=np.float32)

        for i in range(num_agents_in_pool):
            params_i = _get_params_i(pool_params, i)
            for j in range(num_agents_in_pool):
                params_j = _get_params_i(pool_params, j)
                rng, eval_rng = jax.random.split(rng)
                ret_ij = float(evaluate_one_order_jit(eval_rng, params_i, params_j))
                xp_matrix[i, j] = ret_ij
                print(f"XP[{i:02d}, {j:02d}] = {ret_ij:.3f}")

        return xp_matrix

    def build_similarity_matrix(xp_matrix, clamp=True, eps=1e-4):
        """
        Paper-aligned similarity:
            s(i,j) = (XP[i,j] + XP[j,i]) / (XP[i,i] + XP[j,j])

        Notes:
        - If denominator == 0 and numerator == 0, set similarity = 1.
        - Optionally clamp to [0, 1].
        - Add eps to avoid singular matrix issues in spectral clustering.

        Returns a symmetric matrix.
        """
        n = xp_matrix.shape[0]
        sim = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                num = xp_matrix[i, j] + xp_matrix[j, i]
                den = xp_matrix[i, i] + xp_matrix[j, j]

                if abs(den) < 1e-8:
                    s = 1.0 if abs(num) < 1e-8 else 0.0
                else:
                    s = num / den

                if clamp:
                    s = float(np.clip(s, 0.0, 1.0))

                sim[i, j] = s

        # Strong diagonal
        np.fill_diagonal(sim, 1.0)

        # Small positive offset, matching the practical trick described in the paper appendix
        sim = sim + eps
        np.fill_diagonal(sim, 1.0)

        return sim

    def cluster_similarity_matrix(similarity_matrix, num_clusters=None):
        """
        Run spectral clustering on the similarity matrix.
        """
        n = similarity_matrix.shape[0]

        if num_clusters is None:
            # simple fallback heuristic
            num_clusters = min(3, n)
            num_clusters = max(2, num_clusters) if n > 1 else 1

        clustering = SpectralClustering(
            n_clusters=num_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=eval_config.get("SEED", 0),
        )
        cluster_ids = clustering.fit_predict(similarity_matrix)
        return cluster_ids

    def sort_by_cluster(matrix, cluster_ids):
        order = np.argsort(cluster_ids)
        return matrix[order][:, order], order

    def evaluator(rng, do_clustering=True):
        xp_matrix = build_xp_matrix(rng)
        similarity_matrix = build_similarity_matrix(xp_matrix)

        out = {
            "xp_matrix": xp_matrix,
            "similarity_matrix": similarity_matrix,
            "checkpoint_names": pool_names,
        }

        if do_clustering:
            num_clusters = eval_config.get("NUM_CLUSTERS", None)
            cluster_ids = cluster_similarity_matrix(similarity_matrix, num_clusters=num_clusters)

            xp_sorted, order = sort_by_cluster(xp_matrix, cluster_ids)
            sim_sorted, _ = sort_by_cluster(similarity_matrix, cluster_ids)

            out.update({
                "cluster_ids": cluster_ids,
                "cluster_order": order,
                "xp_matrix_sorted": xp_sorted,
                "similarity_matrix_sorted": sim_sorted,
                "checkpoint_names_sorted": [pool_names[i] for i in order],
            })

        return out

    return evaluator


def plot_heatmap(
    matrix,
    title,
    names=None,
    figsize=(7, 6),
    cmap="viridis",
    vmin=None,
    vmax=None,
    save_path=None,
    show=True,
):
    plt.figure(figsize=figsize)
    plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()

    if names is not None and len(names) <= 30:
        plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
        plt.yticks(range(len(names)), names, fontsize=8)
    else:
        plt.xticks(range(matrix.shape[1]))
        plt.yticks(range(matrix.shape[0]))

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_xp_and_similarity(results, save_dir=None, show=True):
    xp = results["xp_matrix"]
    sim = results["similarity_matrix"]
    names = results["checkpoint_names"]

    xp_path = None if save_dir is None else os.path.join(save_dir, "xp_matrix.png")
    sim_path = None if save_dir is None else os.path.join(save_dir, "similarity_matrix.png")

    plot_heatmap(
        xp,
        title="XP Matrix",
        names=names,
        cmap="magma",
        save_path=xp_path,
        show=show,
    )
    plot_heatmap(
        sim,
        title="Similarity Matrix",
        names=names,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        save_path=sim_path,
        show=show,
    )

    if "xp_matrix_sorted" in results:
        xp_sorted_path = None if save_dir is None else os.path.join(save_dir, "xp_matrix_sorted.png")
        sim_sorted_path = None if save_dir is None else os.path.join(save_dir, "similarity_matrix_sorted.png")

        plot_heatmap(
            results["xp_matrix_sorted"],
            title="XP Matrix (sorted by cluster)",
            names=results["checkpoint_names_sorted"],
            cmap="magma",
            save_path=xp_sorted_path,
            show=show,
        )
        plot_heatmap(
            results["similarity_matrix_sorted"],
            title="Similarity Matrix (sorted by cluster)",
            names=results["checkpoint_names_sorted"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            save_path=sim_sorted_path,
            show=show,
        )


def save_xp_results(results, save_path="xp_similarity_results.npz"):
    save_dict = {
        "xp_matrix": results["xp_matrix"],
        "similarity_matrix": results["similarity_matrix"],
        "checkpoint_names": np.array(results["checkpoint_names"], dtype=object),
    }

    if "cluster_ids" in results:
        save_dict["cluster_ids"] = results["cluster_ids"]
        save_dict["cluster_order"] = results["cluster_order"]
        save_dict["xp_matrix_sorted"] = results["xp_matrix_sorted"]
        save_dict["similarity_matrix_sorted"] = results["similarity_matrix_sorted"]
        save_dict["checkpoint_names_sorted"] = np.array(results["checkpoint_names_sorted"], dtype=object)

    np.savez(save_path, **save_dict)
    print(f"Saved to {save_path}")


def run_pair_episode_with_states(config, params_a, params_b, rng):
    """
    Roll out one episode for a fixed pair of agents and store env states for visualization.

    Returns:
        {
            "episode_return": float,
            "state_seq": tree of states stacked over time,
            "reward_seq": np.ndarray [T],
            "done_seq": np.ndarray [T],
            "action_seq_a": np.ndarray [T, num_envs] or [T],
            "action_seq_b": np.ndarray [T, num_envs] or [T],
        }
    """
    eval_config = dict(config)
    eval_config["NUM_ENVS"] = 1  # visualization = one episode / one env

    env = jaxmarl.make(eval_config["ENV_NAME"], **eval_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    network = ActorCriticRNN(
        env.action_space(env.agents[0]).n,
        config=eval_config,
        rnn_type="gru",
    )

    num_eval_steps = eval_config.get("EVAL_NUM_STEPS", eval_config["NUM_STEPS"])

    rng, reset_rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng)

    hstate_a = ScannedRNN.initialize_carry(eval_config, rnn_type="gru", batch_size=1)
    hstate_b = ScannedRNN.initialize_carry(eval_config, rnn_type="gru", batch_size=1)
    done_batch = jnp.zeros((1,), dtype=bool)
    ep_return = jnp.zeros((1,), dtype=jnp.float32)

    def _step_fn(carry, _):
        obs, env_state, hstate_a, hstate_b, done_batch, ep_return, rng = carry
        rng, step_rng = jax.random.split(rng)

        obs_a = obs[env.agents[0]][None, ...]   # (1, H, W, C)
        obs_b = obs[env.agents[1]][None, ...]   # (1, H, W, C)

        ac_in_a = (obs_a[None, ...], done_batch[None, ...])   # (T=1, B=1, H, W, C), (T=1, B=1)
        ac_in_b = (obs_b[None, ...], done_batch[None, ...])

        hstate_a, pi_a, _ = network.apply(params_a, hstate_a, ac_in_a)
        hstate_b, pi_b, _ = network.apply(params_b, hstate_b, ac_in_b)

        action_a = jnp.argmax(pi_a.logits, axis=-1).squeeze()   # scalar
        action_b = jnp.argmax(pi_b.logits, axis=-1).squeeze()   # scalar

        env_act = {
            env.agents[0]: action_a,
            env.agents[1]: action_b,
        }

        next_obs, next_env_state, reward, done, info = env.step(step_rng, env_state, env_act)

        reward_team = jnp.array(
            [0.5 * (reward[env.agents[0]] + reward[env.agents[1]])],
            dtype=jnp.float32,
        )
        new_done_batch = jnp.array([done["__all__"]], dtype=bool)
        ep_return = ep_return + reward_team

        transition_info = {
            "state": env_state.env_state,
            "reward": reward_team[0],
            "done": done["__all__"],
            "action_a": action_a,
            "action_b": action_b,
        }

        new_carry = (
            next_obs,
            next_env_state,
            hstate_a,
            hstate_b,
            new_done_batch,
            ep_return,
            rng,
        )
        return new_carry, transition_info

    init_carry = (
        obsv,
        env_state,
        hstate_a,
        hstate_b,
        done_batch,
        ep_return,
        rng,
    )

    final_carry, traj = jax.lax.scan(_step_fn, init_carry, None, length=num_eval_steps)

    # Also append final state if you want one extra frame at the end
    final_env_state = final_carry[1].env_state

    episode_return = float(final_carry[5].mean())

    state_seq = traj["state"]
    reward_seq = np.array(traj["reward"]).squeeze(-1)
    done_seq = np.array(traj["done"]).squeeze(-1)
    action_seq_a = np.array(traj["action_a"]).squeeze(-1) if np.array(traj["action_a"]).ndim > 1 else np.array(traj["action_a"])
    action_seq_b = np.array(traj["action_b"]).squeeze(-1) if np.array(traj["action_b"]).ndim > 1 else np.array(traj["action_b"])

    return {
        "episode_return": episode_return,
        "state_seq": state_seq,
        "final_state": final_env_state,
        "reward_seq": reward_seq,
        "done_seq": done_seq,
        "action_seq_a": action_seq_a,
        "action_seq_b": action_seq_b,
    }

def save_episode_mp4(
    state_seq,
    save_path,
    agent_view_size=None,
    fps=4,
    tile_size=32,
):
    """
    Save a rendered episode to mp4.

    state_seq: stacked env states over time
    save_path: e.g. "outputs/episode_i0_j1.mp4"
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    viz = OvercookedV2Visualizer(tile_size=tile_size)
    frame_seq = viz.render_sequence(state_seq, agent_view_size=agent_view_size)

    # convert JAX array -> numpy uint8
    frame_seq = np.array(frame_seq).astype(np.uint8)

    with imageio.get_writer(save_path, fps=fps) as writer:
        for frame in frame_seq:
            writer.append_data(frame)

    print(f"Saved mp4 to {save_path}")

def get_pool_params_i(pool_params, i):
    return jax.tree_util.tree_map(lambda x: x[i], pool_params)

def save_pool_pair_episode_mp4(
    config,
    eval_pool,
    i,
    j,
    save_path,
    rng=None,
    fps=4,
    tile_size=32,
):
    if rng is None:
        rng = jax.random.PRNGKey(config["SEED"] + 9999)

    params_i = get_pool_params_i(eval_pool["params"], i)
    params_j = get_pool_params_i(eval_pool["params"], j)

    episode = run_pair_episode_with_states(config, params_i, params_j, rng)

    save_episode_mp4(
        episode["state_seq"],
        save_path=save_path,
        fps=fps,
        tile_size=tile_size,
    )

    print(f"Episode return for pair ({i}, {j}): {episode['episode_return']:.3f}")
    return episode

def save_example_videos_from_results(
    config,
    eval_pool,
    results,
    save_dir,
    fps=4,
    tile_size=32,
):
    os.makedirs(save_dir, exist_ok=True)
    xp = results["xp_matrix"]
    n = xp.shape[0]

    # best off-diagonal pair
    xp_offdiag = xp.copy()
    np.fill_diagonal(xp_offdiag, -np.inf)
    best_idx = np.unravel_index(np.argmax(xp_offdiag), xp_offdiag.shape)

    # worst off-diagonal pair
    xp_offdiag_min = xp.copy()
    np.fill_diagonal(xp_offdiag_min, np.inf)
    worst_idx = np.unravel_index(np.argmin(xp_offdiag_min), xp_offdiag_min.shape)

    # best self-play
    best_self = int(np.argmax(np.diag(xp)))

    base_seed = config["SEED"] + 5000

    print("Saving best off-diagonal video:", best_idx)
    save_pool_pair_episode_mp4(
        config,
        eval_pool,
        i=best_idx[0],
        j=best_idx[1],
        save_path=os.path.join(save_dir, f"best_pair_{best_idx[0]}_{best_idx[1]}.mp4"),
        rng=jax.random.PRNGKey(base_seed + 1),
        fps=fps,
        tile_size=tile_size,
    )

    print("Saving worst off-diagonal video:", worst_idx)
    save_pool_pair_episode_mp4(
        config,
        eval_pool,
        i=worst_idx[0],
        j=worst_idx[1],
        save_path=os.path.join(save_dir, f"worst_pair_{worst_idx[0]}_{worst_idx[1]}.mp4"),
        rng=jax.random.PRNGKey(base_seed + 2),
        fps=fps,
        tile_size=tile_size,
    )

    print("Saving best self-play video:", best_self)
    save_pool_pair_episode_mp4(
        config,
        eval_pool,
        i=best_self,
        j=best_self,
        save_path=os.path.join(save_dir, f"best_selfplay_{best_self}.mp4"),
        rng=jax.random.PRNGKey(base_seed + 3),
        fps=fps,
        tile_size=tile_size,
    )

@hydra.main(version_base=None, config_path="config/oc_extended", config_name="")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    layout_name = config["ENV_KWARGS"]["layout"]
    save_dir = config.get("SAVE_DIR", f"./results/xp_similarity/{layout_name}")
    os.makedirs(save_dir, exist_ok=True)

    final_step = 220
    eval_partner_idx = [i for i in range(0, 15)]
    config["FCP_EVAL_CHECKPOINTS"] = []
    for p_idx in eval_partner_idx:
        config["FCP_EVAL_CHECKPOINTS"].append(
            f"baseline_seed_{p_idx}_step_{final_step}.msgpack"
        )

    # build dummy params exactly as before
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

    # load a pool
    eval_pool = load_partner_pool(config, config["FCP_EVAL_CHECKPOINTS"], dummy_params)

    # make evaluator
    xp_evaluator = make_zsc_evaluator(config, eval_pool)

    # run
    results = xp_evaluator(jax.random.PRNGKey(config["SEED"] + 123), do_clustering=True)

    print("XP matrix shape:", results["xp_matrix"].shape)
    print("Similarity matrix shape:", results["similarity_matrix"].shape)

    if "cluster_ids" in results:
        print("Cluster ids:", results["cluster_ids"])
        for name, cid in zip(results["checkpoint_names"], results["cluster_ids"]):
            print(name, "-> cluster", cid)

    # save
    save_xp_results(results, save_path=os.path.join(save_dir, "xp_similarity_results.npz"))


    # save figures
    plot_xp_and_similarity(
        results,
        save_dir=save_dir,
        show=False,
    )

    # save videos
    video_dir = os.path.join(save_dir, "vids")
    os.makedirs(video_dir, exist_ok=True)
    save_example_videos_from_results(
        config,
        eval_pool,
        results,
        save_dir=video_dir,
        fps=4,
        tile_size=32,
    )


if __name__ == "__main__":
    main()
