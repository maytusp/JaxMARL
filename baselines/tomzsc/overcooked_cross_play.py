import os
import re 
import json 
import hydra 
import chex
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from typing import Any, Tuple 

from baselines.tomzsc.models import JaxMARLLSTM
from baselines.tomzsc.utils.wrappers import CTRolloutManager, load_params
from baselines.tomzsc.ctom import get_ctom
from baselines.tomzsc.utils.utils import *

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    rewards: chex.Array
    done: chex.Array
    avail_actions: chex.Array
    q_vals: chex.Array
    concepts_available: chex.Array = None
    ground_truth_tom: chex.Array = None 
    concepts: chex.Array = None 
    tom_outputs: chex.Array = None 
    state: Any = None 
    info: Any = None


def rollout(config, env, teammate_params, key, ego_params=None,num_envs=1,rewards_only=False,epsilon=0) -> Tuple[Transition,jnp.ndarray]:
    'Simple rollout, e.g. to calculate cross-play or self-play returns'
    wrapped_env = CTRolloutManager(
        env, batch_size=num_envs, preprocess_obs=False
    )
    
    teammate_network = JaxMARLLSTM(
        action_dim=wrapped_env.max_action_space,
        norm_type=config.get("NORM_TYPE","layer_norm"),
        norm_input=config.get("NORM_INPUT",False),
        use_lstm="teammate" in config.get("USE_LSTM",[])
    )

    ego_network = JaxMARLLSTM(
        action_dim=wrapped_env.max_action_space,
        norm_type=config.get("NORM_TYPE","layer_norm"),
        norm_input=config.get("NORM_INPUT",False),
        use_lstm="ego" in config.get("USE_LSTM",[])
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):

        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking

        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        # pick random actions from the valid actions
        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,  # pick the actions that should be random
            random_actions,
            greedy_actions,
        )
        return chosed_actions


    def _step_fn(carry,unused):
        env_state, last_obs, (teammate_hs,ego_hs), last_done, key = carry 
        key,k1,k2,k3,k4,k5 = jax.random.split(key,6)
        _obs = batchify(last_obs)[:,None]
        metrics = {}
        teammate_hs,q_vals= jax.vmap(teammate_network.apply, in_axes=(0, 0,0,None, None))(
            teammate_params, teammate_hs, _obs, last_done[None], False
        )  # (num_agents, num_envs, num_actions)
        q_vals = q_vals.squeeze(1)

        valid_actions = wrapped_env.get_valid_actions(env_state.env_state)
        
        ks = jax.random.split(k1, env.num_agents)
        actions = jax.vmap(eps_greedy_exploration,in_axes=(0,0,None,0))(ks, q_vals, epsilon, batchify(valid_actions))
        actions = unbatchify(actions)
        if ego_params is not None: 
            ego_hs,ego_q = jax.vmap(ego_network.apply, in_axes=(0, 0,0,None))(
                ego_params,ego_hs,_obs,last_done[None]
            )
            ego_q = ego_q.squeeze(1)

            ks = jax.random.split(k3, env.num_agents)
            teacher_actions = jax.vmap(eps_greedy_exploration,in_axes=(0,0,None,0))(ks, ego_q, epsilon, batchify(valid_actions))
            teacher_actions = unbatchify(teacher_actions)
            for i in config["EGO"]:
                actions[f"agent_{i}"]=teacher_actions[f"agent_{i}"] 

        obs, next_env_state, rewards, dones, infos = wrapped_env.batch_step(
            k2, env_state, actions
        )
        transition = Transition(
                    obs=last_obs,               # (num_agents, num_envs, obs_shape)
                    action=actions,             # (num_agents, num_envs,)
                    rewards=rewards["__all__"], # (num_envs,)
                    done=dones["__all__"],      # (num_envs,)
                    avail_actions=None,         # (num_agents, num_envs, num_actions)
                    q_vals=unbatchify(q_vals),  # (num_agents, num_envs, num_actions),
                    state=env_state.env_state,
                    info=metrics
                )
        # rewards_only = don't return transitions to save memory
        ret = (None,infos) if rewards_only else (transition,infos)
        return (next_env_state,obs,(teammate_hs,ego_hs),dones["__all__"],key),ret

    key,k1,k2,k3 = jax.random.split(key,4)
    obs,state = wrapped_env.batch_reset(k1) 
    ego_hs = JaxMARLLSTM.initialize_carry(ego_network.hidden_size,len(env.agents),num_envs)
    teammate_hs = JaxMARLLSTM.initialize_carry(teammate_network.hidden_size,len(env.agents),num_envs)
    _,(transitions,infos) = jax.lax.scan(_step_fn,(state,obs,(teammate_hs,ego_hs),jnp.zeros((num_envs,),dtype=jnp.bool_),key),length=400)
    if transitions is not None:
        _process_trajectory,get_concept_loss,_ = get_ctom(env,config)
        transitions = _process_trajectory(transitions)
    episode_returns = jnp.nanmean(
                        jnp.where(
                            infos["returned_episode"],
                            infos["returned_episode_returns"],
                            jnp.nan,
                        ),
                        axis=(0,-1)
                    )
    return transitions,episode_returns 


def get_transitions(config, key, teammate_params, num_envs, mode="none", track=False, n_random=0, rewards_only=False, epsilon=0, ego_params=None) -> Tuple[Transition,jnp.ndarray]:
    """Do many episodes of cross play or self play.
    
    teammate_params (dict): A set of network parameters, such as for doing self-play
    num_envs (int): Number of parallel environments
    mode (str): Determines how agents are paired up. 
        - "none": Do self-play, with num_envs x n_teammates total rollouts
        - "cross_play": Do cross-play, with num_envs x n_teammates x n_ego total rollouts if ego_params is specified, or num_envs x n_teammates x n_teammates otherwise.
        - "random": Randomly pair up teammate with ego (or teammate with teammate if ego_params is None), for num_envs x n_random total rollouts. Used for example to roll out a cluster Best-Response with teammates from that cluster
    n_random (int): When mode is "random", how many iters to scan
    rewards_only (bool): Only return scalar rewards, no transitions, to save space
    ego_params (dict): Another set of parameters, such as for doing cross-play
    """
    key,k1,k2,k3,k4 = jax.random.split(key,5)
    env,_ = env_from_config(config)

    # Params can be a list of paths or a vectorized parameter dictionary
    if isinstance(teammate_params, list) and all(isinstance(p, str) for p in teammate_params):
        teammate_params = [tree_batchify(load_params(p)) for p in teammate_params]
        teammate_params = jax.tree.map(lambda *args:jnp.stack(args),*teammate_params)
    
    if ego_params is not None:
        if isinstance(ego_params, list) and all(isinstance(p, str) for p in ego_params):
            ego_params = [tree_batchify(load_params(p)) for p in ego_params]
            ego_params = jax.tree.map(lambda *args:jnp.stack(args),*ego_params)
        n_ego = get_batch_size(ego_params)
        
    n_teammates = get_batch_size(teammate_params)
    permk = jax.random.split(k1,n_teammates) 

    # Pair up parameters from a batched parameter dictionary so that they're suitable for cross-play
    # Then flatten so that we can scan over it 
    if mode=="cross_play":
        # (n_teammates, 2) -> (n_teammates, n_teammates, 2)
        if ego_params is None:
            def make_xplay(x):
                i,j = jnp.meshgrid(jnp.arange(x.shape[0]),jnp.arange(x.shape[0]),indexing="ij") 
                x = jnp.stack([x[i,0],x[j,1]],axis=2) 
                return x.reshape([-1,*x.shape[2:]])
            teammate_params = jax.tree.map(make_xplay,teammate_params)
            permk = jax.vmap(lambda x:jax.random.split(x,n_teammates))(permk)
            permk = permk.reshape((-1,*permk.shape[2:]))
        else:
            # (n_teammates, 2) and (n_ego, 2) -> (n_teammates, n_ego, 2) and (n_teammates, n_ego, 2)
            i,j = jnp.meshgrid(jnp.arange(n_teammates),jnp.arange(n_ego),indexing="ij")
            ego_params = jax.tree.map(lambda x:x[i].reshape([-1,*x.shape[1:]]),ego_params)
            teammate_params = jax.tree.map(lambda x:x[j].reshape([-1,*x.shape[1:]]),teammate_params)
            permk = jax.vmap(lambda x:jax.random.split(x,n_ego))(permk)
            permk = permk.reshape((-1,*permk.shape[2:]))
    elif mode=="random":
        # (n_teammates, 2) -> (n_random, 2)
        if ego_params is None:
            i, j = jax.random.choice(k2,n_teammates,(n_random*2,)).reshape([2,-1]) 
            def getij(i,j):
                i = jax.tree.map(lambda x:x[i,0],teammate_params) 
                j = jax.tree.map(lambda x:x[j,1],teammate_params) 
                return tree_batchify({"agent_0":i,"agent_1":j}) 
            teammate_params = jax.vmap(getij)(i,j) 
        # (n_teammates, 2) and (n_ego, 2) -> (n_random, 2) and (n_random, 2)
        else:
            i = jax.random.choice(k2,n_teammates,(n_random,))
            j = jax.random.choice(k3,n_ego,(n_random,))
            teammate_params = jax.tree.map(lambda x:x[i],teammate_params)
            ego_params = jax.tree.map(lambda x:x[j],ego_params)
        permk = jax.random.split(k1,n_random)
    
    # Do a single rollout for the given teammate and ego (may be None) parameters
    def get_single(k,p,tp):
        k,k1,k2 = jax.random.split(k,3)
        transitions,returns = rollout(config,env,p,k2,ego_params=tp,num_envs=num_envs,rewards_only=rewards_only,epsilon=epsilon)
        return transitions,returns
    
    # Get transitions and returns -> (n_envs, n_pairs, n_steps)
    s = 1
    transitions,returns = scan_loop(get_single,s=s,track=track)(permk,teammate_params,ego_params)

    
    transitions = jax.tree.map(lambda x:x.swapaxes(1,2),transitions) # reshape -> n_pairs, n_envs, n_steps
    
    # Unflatten
    if mode=="cross_play":
        if ego_params is None:
            transitions,returns  = jax.tree.map(lambda x:x.reshape([n_teammates,n_teammates,*x.shape[1:]]),(transitions,returns))
        else:
            transitions,returns  = jax.tree.map(lambda x:x.reshape([n_ego,n_teammates,*x.shape[1:]]),(transitions,returns))
    elif mode=="random":
        transitions = jax.tree.map(lambda x:jnp.reshape(x,[-1,*x.shape[2:]]),transitions)

    return transitions,returns 


def do_cross_play_eval(config: dict):
    mode = config["CROSS_PLAY_MODE"]
    epsilon = config["EPSILON"]

    # Get teammate and maybe ego paths
    teammate_paths, ego_paths, _, cluster_labels = get_param_paths(config)

    # If grouped into clusters, run random pairings within/across clusters. 
    # Because the number of teammate teams in each cluster can vary, we need to do this manually here rather than vectorizing it in get_transitions (very slow due to repeated jit)
    if cluster_labels is not None:
        if mode == "random":
            returns = [get_transitions(config,jax.random.key(12),tp,1,mode="random",n_random = 100, ego_params=[ep],rewards_only=True,epsilon=epsilon, track=True)[1] for ep,tp in zip(ego_paths, teammate_paths)]
            returns = jnp.stack(returns)
            returns = returns.reshape([len(ego_paths), -1])
        elif mode == "cross_play":
            returns = [[get_transitions(config,jax.random.key(12),tp,1,mode="random",n_random = 100, ego_params=[ep],rewards_only=True,epsilon=epsilon, track=True)[1] for tp in teammate_paths] for ep in ego_paths] 
            returns = jnp.stack([jnp.stack(r) for r in returns]) # n_ego, n_teammate
            returns = returns.reshape([len(ego_paths), len(teammate_paths), -1])
        else:
            raise ValueError(f"Invalid cross play mode for when cluster labels are specified {mode}")
    # Self-play, cross-play, random pairings, etc.
    else:
        # 25 envs avoids OOM for 10x10 vectorized rollouts
        _,returns = get_transitions(config,jax.random.key(12),teammate_paths,1,mode=mode,ego_params=ego_paths,rewards_only=True,epsilon=epsilon, track=True)

    # Average/std over parallel environments and save
    alg_name = config.get("ALG_NAME", "cross_play")
    _, env_name = env_from_config(config)
    save_dir = os.path.join("overcooked_cache/cross_play", f'{alg_name}_{env_name}')

    os.makedirs(save_dir,exist_ok=True)
    with open(os.path.join(save_dir, "avg.json"),"w") as f:
        json.dump(returns.mean(axis=-1).tolist(),f)
    with open(os.path.join(save_dir, "std.json"),"w") as f:
        json.dump(returns.std(axis=-1).tolist(),f)
    with open(os.path.join(save_dir, "max.json"),"w") as f:
        json.dump(returns.max(axis=-1).tolist(),f)
    with open(os.path.join(save_dir, "min.json"),"w") as f:
        json.dump(returns.min(axis=-1).tolist(),f)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    config = {**config, **config["alg"]}

    env, _ = env_from_config(config)
    if set(config["EGO"]+config.get("TEAMMATE",[])) != set(range(env.num_agents)):
        raise ValueError("Agents must either be classified as EGO or TEAMMATE")
    if set(config["EGO"]).intersection(config["TEAMMATE"]) != set():
        raise ValueError("Agents cannot be classified as both EGO and TEAMMATE")

    do_cross_play_eval(config)


if __name__ == "__main__":
    main()