"""
Specific to this implementation: CNN network and Reward Shaping Annealing as per Overcooked paper.
"""
import os
import json 
import hydra 
import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any
from functools import partial
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf
from flax.training.train_state import TrainState

from utils.utils import *
from utils.wrappers import (
    CTRolloutManager,
    load_params
)

from ctom import get_ctom, bce, bkl
from models import OvercookedPredictConcepts, JaxMARLLSTM, JaxMARLLSTMGCRL

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

class CustomTrainState(TrainState):
    batch_stats: Any

def run_overcooked_single(config, env, teammate_params, ego_params, key, cluster_tom_params=None, global_tom_params=None, rewards_only=False, env_name = None, training_params = None, cross_play_stats = None):
    num_envs = 1

    method = config["METHOD"]

    # Number of each kind of agent (when the "ego" agent during evaluation is an ensemble, we count the number of agents in ensemble)
    num_teammates = get_batch_size(teammate_params)
    num_ego = get_batch_size(ego_params)

    cluster_labels = config.get("CLUSTER_LABELS") # No clustering = everyone is in own cluster
    cluster_labels = cluster_labels or list(range(num_ego))
    n_clusters = int(max(cluster_labels))+1

    # Setup ego and teammate networks
    wrapped_env = CTRolloutManager(
        env, batch_size=num_envs, preprocess_obs=False
    )

    teammate_network = JaxMARLLSTM(
        action_dim=wrapped_env.max_action_space,
        norm_type=config.get("NORM_TYPE","layer_norm"),
        norm_input=config.get("NORM_INPUT",False),
        use_lstm="teammate" in config.get("USE_LSTM",[])
    )

    if method == "pecan":
        ego_network = JaxMARLLSTMGCRL(
            action_dim=wrapped_env.max_action_space,
            num_concepts = n_clusters+1,
            norm_type=config["NORM_TYPE"],
            norm_input=config["NORM_INPUT"],
            use_lstm="ego" in config.get("USE_LSTM",[])
        )
    else:
        ego_network = JaxMARLLSTM(
            action_dim=wrapped_env.max_action_space,
            norm_type=config.get("NORM_TYPE","layer_norm"),
            norm_input=config.get("NORM_INPUT",False),
            use_lstm="ego" in config.get("USE_LSTM",[])
        )
    
    # Setup ToM models
    _process_trajectory,get_concept_loss,_ = get_ctom(env,config)
    concept_set = config.get("CONCEPT_SET")
    concept_shape, concept_dim = get_concept_shape(concept_set)

    global_tom_uses_lstm = "global_tom" in config.get("USE_LSTM",{})
    cluster_tom_uses_lstm = "cluster_tom" in config.get("USE_LSTM",{})
    global_tom_network  = OvercookedPredictConcepts(concept_dim=concept_dim,concept_shape=concept_shape,use_lstm=global_tom_uses_lstm)
    cluster_tom_network  = OvercookedPredictConcepts(concept_dim=concept_dim,concept_shape=concept_shape,use_lstm=cluster_tom_uses_lstm)
    
    # Assume that we're in a two-player game
    ego_id = config["EGO"][0]
    teammate_id = config["TEAMMATE"][0]
    
    key,k1,k2,k3,k4 = jax.random.split(key,5)

    # Choose a random teammate
    random_agent_index = jax.random.choice(k3,num_teammates) 
    teammate_params = jax.tree.map(lambda x:x[random_agent_index],teammate_params)

    # Select initial ego agent
    if method=="self_play":
        ego_idx = random_agent_index 
    elif method in {"random_selection", "tbs", "cbpr", "pecan", "mesh", "cem"}:
        ego_idx = jax.random.choice(k4, num_ego)
    else:
        raise ValueError(f"No such method {method}")
    
    # Window hyperparameters for TBS
    steps_per_eval = config.get("STEPS_PER_EVAL",1)
    steps_to_keep = config.get("STEPS_TO_KEEP",1)
    initial_steps = config.get("INITIAL_STEPS",0)

    # One environment step
    def step_fn(carry,unused):
        # Setup
        timestep,obs, hidden,history,ego_idx,last_done,env_state,key = carry 
        teammate_hs,ego_hs,cluster_tom_hs,global_tom_hs,training_hs = hidden 
        prior_history,cluster_tom_outputs_history,global_tom_outputs_history, concepts_available_history, concepts_history = history
        key,k1,k2,k3,k4,k5,k6 = jax.random.split(key,7)

        metrics = {}

        # Teammate action
        teammate_hs,teammate_q= jax.vmap(teammate_network.apply, in_axes=(0, 0,0,None, None))(
            teammate_params, teammate_hs, batchify(obs)[:,None,None], last_done[None,None], False
        )  # (num_agents, num_envs, num_actions)
        teammate_q=teammate_q.squeeze(1)

        valid_actions = wrapped_env.get_valid_actions(env_state.env_state)
        actions = jnp.argmax(teammate_q.squeeze(1) - (1-batchify(valid_actions).squeeze(1))*1000,axis=-1)
        actions = unbatchify(actions)
        
        # All policy actions in ego agent ensemble
        ego_hs,ego_output = jax.vmap(jax.vmap(ego_network.apply, in_axes=(0, 0,0,None)),in_axes=(0,0,None,None))(
            ego_params,ego_hs,batchify(obs)[:,None,None],last_done[None,None]
        )
        ego_q = ego_output[0] if method == "pecan" else ego_output
        ego_q = ego_q.squeeze((2,3))
        
        # Ensemble methods
        if method in {"tbs", "cbpr", "mesh", "cem"}:
            if method in {"cbpr", "mesh", "cem"}:
                training_hs,training_q_vals= jax.vmap(jax.vmap(teammate_network.apply, in_axes=(0, 0,0,None, None)),in_axes=(0,0,None,None,None))(
                    training_params, training_hs, batchify(obs)[:,None,None], last_done[None,None], False
                )  # (num_agents, num_envs, num_actions)
                training_q_vals = training_q_vals.squeeze(3)
                training_actions = jnp.argmax(training_q_vals.squeeze(2) - (1-batchify(valid_actions).squeeze(1))*1000,axis=-1) # (2,6) vs (n_params,2,6)
                training_actions = jax.vmap(unbatchify)(training_actions)[f"agent_{teammate_id}"] # (n_params,) # Get P(unknown teammate's action; pi)
                predictor_output = jax.nn.one_hot(training_actions, 6)*0.9 + 0.1/6 # (n_params, 6). Assume 0.1 epsilon-soft policy
                predictor_output = jnp.stack([jnp.stack([predictor_output[i] for i,l in enumerate(cluster_labels) if l==t]).mean(axis=0) for t in range(num_ego)])
            else:
                cluster_tom_hs, predictor_output = jax.vmap(jax.vmap(cluster_tom_network.apply,in_axes=(0,0,0,None)),in_axes=(0,0,None,None))(cluster_tom_params,cluster_tom_hs,batchify(obs)[:,None,None],last_done[None,None])
                predictor_output=predictor_output.squeeze((2,3))
                predictor_output = jax.vmap(unbatchify)(predictor_output)[f"agent_{ego_id}"][...,1,:] # get 2nd order ToM about partner agent
            
            if method in {"cbpr", "mesh", "cem"}:
                observer_output = jnp.zeros_like(global_tom_outputs_history[-1])
            else:
                global_tom_hs, observer_output = jax.vmap(global_tom_network.apply,in_axes=(0,0,0,None))(global_tom_params,global_tom_hs,batchify(obs)[:,None,None],last_done[None,None])
                observer_output = unbatchify(observer_output)[f"agent_{ego_id}"].squeeze((0,1))[...,1,:] # get 1st order ToM about partner agent
            
            _transition = Transition(obs=None,action=None,rewards=jnp.float32(0),done=last_done,avail_actions=None,q_vals=None,state=env_state.env_state,tom_outputs=unbatchify(observer_output))
            _transition = jax.tree.map(lambda x:x[None,None],_transition) 
            _transition = _process_trajectory(_transition) 
            _transition = jax.tree.map(lambda x:x.squeeze((0,1)),_transition)
            concepts = unbatchify(_transition.concepts)[f"agent_{teammate_id}"][...,0,:]

            # Methods based on modeling low level actions
            if method in {"cbpr", "mesh", "cem"} or concept_set=="actions":
                concepts = jax.nn.one_hot(actions[f"agent_{teammate_id}"], 6) # num_agents, num_envs, num_steps, num_actions
            # For concept set ablation experiments
            if concept_set in {"coarse", "super_coarse"}:
                concepts = construct_coarse_concepts(concepts, super_coarse = (concept_set == "super_coarse"))

            concepts_history = jnp.roll(concepts_history,-1,axis=0).at[-1].set(concepts)
            cluster_tom_outputs_history = jnp.roll(cluster_tom_outputs_history,-1,axis=0).at[-1].set(predictor_output)
            global_tom_outputs_history = jnp.roll(global_tom_outputs_history,-1,axis=0).at[-1].set(observer_output)
            concepts_available_history = jnp.roll(concepts_available_history,-1,axis=0).at[-1].set(1)

            def backprop_concepts(concepts,avail,pred):
                def scan_fn(prev_concept,x):
                    concept,avail,pred = x 
                    new_concept = jax.lax.cond(jnp.sum(concept)>0,lambda:concept,lambda:prev_concept)
                    new_concept = jax.lax.cond(avail>0,lambda:new_concept,lambda:jnp.zeros_like(new_concept))
                    output = jax.lax.cond(jnp.sum(new_concept)>0,lambda:new_concept,lambda:pred)
                    return new_concept,(output,new_concept) 
                _,(concepts,new_concept) = jax.lax.scan(scan_fn,jnp.zeros(concepts.shape[1:],dtype=concepts.dtype),(concepts,avail,pred),reverse=True)
                return (concepts,new_concept) 
            
            concepts,predictor_output,avail = concepts_history[-steps_to_keep:],cluster_tom_outputs_history[-steps_to_keep:],concepts_available_history[-steps_to_keep:]
            observer_output = jnp.broadcast_to(observer_output,global_tom_outputs_history[-steps_to_keep:].shape)
            (tom_target,new_concept) = backprop_concepts(concepts,avail,observer_output)

            # Bayesian update
            if method == "cbpr": # tom_target is (n_timesteps, n_actions), predictor_output is (n_timesteps, n_teachers, n_actions), avail is (n_timesteps)
                pi = (predictor_output * tom_target[:,None]).sum(axis=-1) # (n_timesteps, n_teachers) - pi_bar(opponent_action)
                pi = jnp.log(jnp.where((avail==1)[:,None], pi, 1))[-steps_to_keep:]
                
                p_q_given_tau = jnp.exp(pi.sum(axis=0)) # (n_teachers,) - P(q | tau)
                xi_tau = prior_history = (p_q_given_tau*prior_history) / jnp.sum(p_q_given_tau*prior_history) # (n_teachers,) intra-episode belief
                rho_t = 0.**jnp.sum(avail) # rho^t 
                
                zeta_tau = rho_t/num_ego + (1-rho_t)*xi_tau # (n_teachers,) weighted average of inter-episode belief (uniform prior) and intra-episode belief
                u_max = jnp.max(cross_play_stats["max"]) # (,) # cross_play_stats is (n_teachers, n_teachers) as (br_id,cluster_id). zeta_tau is (n_teachers)
                u_avg = jnp.max((cross_play_stats["avg"]*zeta_tau).sum(axis=-1)) # (,) # max over best responses of the average reward over teammate cluster ids.
                
                get_cdf = jax.vmap(jax.vmap(jax.scipy.stats.norm.cdf, in_axes=(None,)),in_axes=(None,)) # double vmap over loc and scale, no vmap over u_max and u_avg
                upper = get_cdf(u_max, loc = cross_play_stats["avg"], scale = cross_play_stats["std"]) # (n_teachers, n_teachers) as (br_id, cluster_id)
                lower = get_cdf(u_avg, loc = cross_play_stats["avg"], scale = cross_play_stats["std"]) # (n_teachers, n_teachers) as (br_id, cluster_id)
                integral = upper - lower # integral[pi,tau] = integral from u_avg to u_max of P(U|tau,pi). Since U|tau,pi is normal, we do CDF(u_max)-CDF(u_avg)
                utility = (integral*zeta_tau).sum(axis=-1) # utility[pi] = Sum over tau of zeta_tau*integral[pi,tau]
                new_ego_idx = random_argmin(k4, -utility) # chosen policy is the one with the highest utility.
                
            # Distance based selection
            elif method == "mesh":
                d = predictor_output.max(axis=-1)-(predictor_output * tom_target[:,None]).sum(axis=-1) # (n_timesteps, n_teachers) probability difference between argmax and actual
                factor = (0.99)**(d.shape[0]-jnp.arange(d.shape[0]))*avail # discount factor
                d = (d*factor[:,None]).sum(axis=0) # Sum of discounted distances, (n_teachers, )
                d_tilde = d/d.sum() # Normalized distances
                w = (1-d_tilde)/(1-d_tilde).sum() # Normalized distances
            else:
                predictor_losses = jax.vmap(partial(bce,reduction="none"),in_axes=(1,None))(predictor_output*0.999+0.0005,tom_target*0.999+0.0005)
                predictor_losses = predictor_losses.mean(axis=range(2,predictor_losses.ndim)) 
                total_losses = (predictor_losses*avail).sum(axis=-1)/avail.sum() 
                metrics["total_losses"]=total_losses
                metrics["tom_target"]=tom_target
                metrics["concepts_history"]=concepts
                metrics["next_concepts"]=new_concept
                new_ego_idx = random_argmin(k4,total_losses)

        else:
            new_ego_idx = ego_idx 

        if method == "mesh":
            moe_q = (ego_q*w[:,None,None]).sum(axis=0)
            ego_actions = jnp.argmax(moe_q-(1-batchify(valid_actions).squeeze(1))*1000,axis=-1)
            ego_action = unbatchify(ego_actions)[f"agent_{ego_id}"]
            metrics["correct_action"] = jnp.float32(actions[f"agent_{ego_id}"] == ego_action)
            actions[f"agent_{ego_id}"]=ego_action
            # Mesh does MoE rather than selection
            ego_idx = 0
            reselect =  False
        else:
            reselect = (timestep>=initial_steps) & (((timestep)%steps_per_eval)==0)
            switch_selection = jnp.float32(jnp.where(reselect,ego_idx!=new_ego_idx,False))
            ego_idx = jnp.where(reselect,new_ego_idx,ego_idx)

            ego_actions = jnp.argmax(ego_q-(1-batchify(valid_actions).squeeze(1))*1000,axis=-1)
            ego_actions = jax.vmap(unbatchify)(ego_actions) 
            metrics["correct_action"] = jnp.float32(actions[f"agent_{ego_id}"] == ego_actions[f"agent_{ego_id}"][ego_idx])
            actions[f"agent_{ego_id}"]=ego_actions[f"agent_{ego_id}"][ego_idx]
        
        new_obs, new_env_state, reward, new_done, info = env.step(k1, env_state, actions)

        if method!="mesh":
            info["switch_selection"]=switch_selection
        for k,v in metrics.items():
            info[k]=v

        transition = Transition(
                    obs=obs,  # (num_agents, num_envs, obs_shape)
                    action=actions,  # (num_agents, num_envs,)
                    rewards=reward[f"agent_{ego_id}"],  # (num_envs,)
                    done=new_done["__all__"],  # (num_envs,)
                    avail_actions=None,  # (num_agents, num_envs, num_actions)
                    q_vals=unbatchify(teammate_q),  # (num_agents, num_envs, num_actions),
                    state=env_state.env_state
                )
        
        hidden = (teammate_hs,ego_hs,cluster_tom_hs,global_tom_hs,training_hs)
        history = (prior_history,cluster_tom_outputs_history,global_tom_outputs_history, concepts_available_history,concepts_history)
        return (timestep+1,new_obs,hidden,history,ego_idx,new_done["__all__"],new_env_state,key),(transition,info)
    
    obs,state = env.reset(k1)
    
    # Init hidden states
    teammate_hs = JaxMARLLSTM.initialize_carry(teammate_network.hidden_size,len(env.agents),num_envs)
    ego_cls = JaxMARLLSTMGCRL if method == "pecan" else JaxMARLLSTM
    ego_hs = ego_cls.initialize_carry(ego_network.hidden_size,num_ego,env.num_agents,num_envs)
    cluster_tom_hs = OvercookedPredictConcepts.initialize_carry(cluster_tom_network.rnn_dim,num_ego,env.num_agents,num_envs)
    global_tom_hs = OvercookedPredictConcepts.initialize_carry(global_tom_network.rnn_dim,env.num_agents,num_envs)
    training_hs = JaxMARLLSTM.initialize_carry(teammate_network.hidden_size,get_batch_size(training_params),len(env.agents),num_envs)
    init_hs = (teammate_hs,ego_hs,cluster_tom_hs,global_tom_hs,training_hs) 

    # Methods that need to track history
    if method in {"tbs", "cbpr", "mesh", "cem"}:
        num_hl_actions = concept_shape[-1]
        prior_history = jnp.ones((num_ego,))
        cluster_tom_outputs_history = jnp.zeros((400,num_ego,num_hl_actions)) 
        global_tom_outputs_history = jnp.zeros((400,num_hl_actions)) 
        concepts_available_history = jnp.zeros(400)
        concepts_history = jnp.zeros((400,num_hl_actions))
        history = (prior_history,cluster_tom_outputs_history,global_tom_outputs_history,concepts_available_history,concepts_history)
    else:
        history = (None,None,None,None,None)

    # Run single episode
    _,(transitions,infos) = jax.lax.scan(step_fn,(1,obs,init_hs,history,ego_idx,jnp.bool_(False),state,k2),length=400)
    transitions = jax.tree.map(lambda x:x[:,None],transitions) 
    transitions = _process_trajectory(transitions) 
    transitions = jax.tree.map(lambda x:x.squeeze(1),transitions)
    
    episode_returns = jnp.nanmean(
                        jnp.where(
                            infos["returned_episode"],
                            infos["returned_episode_returns"],
                            jnp.nan,
                        )
                    )
    
    # Report metrics
    _infos = {k:v for k,v in infos.items() if k in ["correct_action", "switch_selection"]}
    res = {"total_returns":episode_returns, **(_infos if config["VERBOSE"] else {})}
    return res if rewards_only else {**res,"transitions":transitions}

def run_overcooked_multiple(config,env,params,ego_params,key,cluster_tom_params=None,global_tom_params=None,n_iter=1,env_name = None, training_params = None, cross_play_stats = None):
    def step_fn(key,unused):
        key,k1 = jax.random.split(key)
        res = run_overcooked_single(config,env,params,ego_params,k1,cluster_tom_params=cluster_tom_params,global_tom_params=global_tom_params,rewards_only=True, env_name = env_name, training_params = training_params, cross_play_stats = cross_play_stats) 
        return key,res 
    _,res = jax.lax.scan(scan_tqdm(n_iter,print_rate=1)(step_fn),key,xs=jnp.arange(n_iter)) 
    res = jax.tree.map(lambda x:x.tolist(),res)
    return res    

def get_overcooked_eval_data(config):
    key = jax.random.key(config["SEED"])
    key,k1,k2,k3,k4 = jax.random.split(key,5)
    env, env_name = env_from_config(config)

    # Load params
    teammate_paths, ego_paths, training_paths, cluster_labels = get_param_paths(config, group_teammates=False)
    config["CLUSTER_LABELS"] = cluster_labels 
    
    teammate_params = load_from_paths(teammate_paths)
    ego_params = load_from_paths(ego_paths)
    training_params = load_from_paths(training_paths)

    cluster_tom_path, global_tom_path = get_tom_paths(config)
    cluster_tom_params = global_tom_params = None 
    if cluster_tom_path is not None:
        cluster_tom_params = load_params(cluster_tom_path)
    if global_tom_path is not None:
        global_tom_params = load_params(global_tom_path)

    cross_play_dir = config.get("CROSS_PLAY_DIR")    
    cross_play_stats = None
    if cross_play_dir is not None:
        cross_play_stats = {}
        for k in ["max", "min", "avg", "std"]:
            with open(os.path.join(cross_play_dir, f"{k}.json"),"r") as f:
                cross_play_stats[k] = json.load(f)
        cross_play_stats = {k:np.array(v) for k,v in cross_play_stats.items()}
    
    # Run evaluation
    n_iters = 300
    return run_overcooked_multiple(
        config, 
        env, 
        teammate_params, 
        ego_params, 
        k2, 
        cluster_tom_params = cluster_tom_params,
        global_tom_params = global_tom_params,
        n_iter= n_iters,
        env_name = env_name,
        training_params = training_params,
        cross_play_stats = cross_play_stats
    )
    



def do_overcooked_eval(config: dict):
    # Train ToM
    res = get_overcooked_eval_data(config)
    
    # Save
    alg_name = config.get("ALG_NAME", f"{config["METHOD"]}_eval")
    _, env_name = env_from_config(config)
    save_dir = os.path.join(f"overcooked_cache/eval/{env_name}")

    os.makedirs(save_dir,exist_ok=True)
    with open(os.path.join(save_dir, f'{alg_name}.json'),"w") as f:
        json.dump(res,f)
    
    
@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    config = {**config, **config["alg"]}

    env, _ = env_from_config(config)
    if not (len(config["EGO"])==1 and len(config["TEAMMATE"])==1 and set(config["EGO"]+config["TEAMMATE"])==set(range(env.num_agents))):
        raise ValueError("Evaluation assumes one ego and one different teammate agent")
    if config.get("EGO_DIR") is None or config.get("TEAMMATE_DIR") is None:
        raise ValueError("Must specify ego and teammate agent for evaluation")
    if config["METHOD"] == "tbs":
        if config.get("TOM_DIR") is None or config.get("CLUSTER_LABELS") is None:
            raise ValueError("Must specify ToM models and cluster labels for TBS")
    if config["METHOD"] == "cbpr" and config.get("CROSS_PLAY_DIR") is None:
        raise ValueError("Must specify cross-play stats for TBS")
    if config["METHOD"] in {"cbpr", "mesh", "cem"} and config.get("TRAINING_DIR") is None:
        raise ValueError("Must specify training pool for the given method")
    if config["METHOD"] == "pecan" and config.get("CLUSTER_LABELS") is None:
        raise ValueError("Must specify cluster labels for pecan")
    
    do_overcooked_eval(config)

if __name__ == "__main__":
    main()