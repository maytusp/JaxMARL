"""
Specific to this implementation: CNN network and Reward Shaping Annealing as per Overcooked paper.
"""
import os
import json 
import optax
import jax
import jax.numpy as jnp
from tqdm import tqdm 
from typing import Any
from functools import partial
from jax_tqdm import scan_tqdm
from flax.training.train_state import TrainState

from baselines.tomzsc.utils.utils import *
from baselines.tomzsc.utils.wrappers import save_params 
from baselines.tomzsc.ctom import bce, make_tom_target
from baselines.tomzsc.models import OvercookedPredictConcepts
from baselines.tomzsc.overcooked_cross_play import *

class CustomTrainState(TrainState):
    batch_stats: Any

def train(config,n_epochs,n_steps,transitions: Transition,key,tom_output=None,use_lstm=True,init_params=None,lr_init=None):
    "Base function for training cluster and global ToM"
    key,k1,k2,k3 = jax.random.split(key,4)
    
    # Initialize ToM model
    concept_set = config.get("CONCEPT_SET",None)
    concept_shape, concept_dim = get_concept_shape(concept_set)
    model = OvercookedPredictConcepts(concept_dim=concept_dim,concept_shape=concept_shape,use_lstm=use_lstm)
    ndata = get_batch_size(transitions)
    obs_shape = transitions.obs["agent_0"].shape[2:] # n_agents, n_envs, n_steps, *obs_shape

    def create_state(key):
        init_x = jnp.zeros((1,1,*obs_shape))
        init_done = jnp.zeros((1,1))
        hidden = OvercookedPredictConcepts.initialize_carry(model.rnn_dim,1) 
        variables = model.init(key,hidden,init_x,init_done)
        lr_scheduler = optax.cosine_decay_schedule(lr_init or 0.001,n_steps*n_epochs)
        tx = optax.chain(
            optax.clip_by_global_norm(10),
            optax.radam(learning_rate=lr_scheduler)
        )
        train_state = CustomTrainState.create(apply_fn=model.apply,params=variables,batch_stats=None,tx=tx)
        return train_state 
    train_state = jax.vmap(create_state)(jax.random.split(k1,2))
    if init_params is not None:
        train_state = train_state.replace(params=init_params)
    
    # Model inputs and ground-truth outputs
    inp = transitions.obs 
    if concept_set == "actions":
        otp = jax.nn.one_hot(batchify(transitions.action), 6) # num_agents, num_envs, num_steps, num_actions
        otp = batchify({"agent_0":otp,"agent_1":otp[::-1]}) # num_agents, num_agents, num_envs, num_steps, num_actions 
        otp = jnp.transpose(otp,(2,3,0,1,4)) # num_envs, num_steps, num_agents, num_agents, num_actions
    elif concept_set in {"coarse", "super_coarse"}:
        otp = construct_coarse_concepts(transitions.ground_truth_tom, super_coarse = (concept_set == "super_coarse"))
    else:
        otp = transitions.ground_truth_tom
        
    # Sampling 
    def get_batch(key,bsz=16):
        data = jax.random.choice(key,ndata,(bsz,))
        if tom_output is not None:
            res = jax.tree.map(lambda x:x[data],(inp,otp,tom_output))
        else:
            res = jax.tree.map(lambda x:x[data],inp),otp[data]
        return res
        
    # Train one iter
    def train_iter(carry,unused,bsz=16):
        train_state,key = carry
        key,k1,k2 = jax.random.split(key,3)
        if tom_output is not None:
            inp,otp,tom = get_batch(k1,bsz=bsz)
        else:
            inp,otp = get_batch(k1,bsz=bsz)
            
        dones = jnp.zeros(jax.tree.leaves(inp)[0].shape[:2])
        otp = otp.swapaxes(0,1)
        inp = jax.tree.map(lambda x:x.swapaxes(0,1),inp)
        if tom_output is not None:
            tom = jax.tree.map(lambda x:x.swapaxes(0,1),tom)
        dones = dones.swapaxes(0,1)

        def loss_fn(params):
            hidden = OvercookedPredictConcepts.initialize_carry(model.rnn_dim,2,bsz) 
            _,out = jax.vmap(model.apply,in_axes=(0,0,0,None))(params,hidden,batchify(inp),dones)
            
            out = jnp.moveaxis(out,0,2)

            make_all_targets = lambda output,gt:jax.vmap(make_tom_target,in_axes=(None,0,0))(output,gt,jnp.arange(2))
            tom_out = out if tom_output is None else tom
            tom_targets = jax.vmap(jax.vmap(lambda o,t:make_all_targets(o,t)))(tom_out,otp) # (num_steps, num_envs, num_agents, num_agents, concept_dim)
            tom_loss = bce(out*0.999+0.0005,jax.lax.stop_gradient(tom_targets)*0.999+0.0005)

            return tom_loss,out

        (loss,out),grads = jax.value_and_grad(loss_fn,has_aux=True)(train_state.params) 
        train_state = jax.vmap(lambda t,g:t.apply_gradients(grads=g))(train_state,grads)

        return (train_state,key),loss 

    # Train
    bsz = min(8,ndata)
    jit_train_iter = jax.jit(scan_tqdm(n_steps)(partial(train_iter,bsz=bsz)),donate_argnums=(0,),device=GPU)
    for i in tqdm(range(n_epochs)):
        print("\n\n") # 2-level tqdm hack
        (train_state,_),losses = jax.lax.scan(jit_train_iter,(train_state,k3),xs=jnp.arange(n_steps),length=n_steps) 
    return train_state,losses


def train_tom_models(config, paths, ego_params=None):
    # Setup
    env,_ = env_from_config(config)
    key = jax.random.key(3)
    key,k1,k2,k3,k4 = jax.random.split(key,5) 

    # Sample cluster BR + same cluster teammate trajectories
    print(f"Sampling intra-cluster trajectories")
    if ego_params is not None:
        res = [get_transitions(config, k4, p, 1, ego_params=[tp]*len(p), mode="random", n_random=250, rewards_only=False, epsilon=0.01, track=True) for p,tp in zip(paths,ego_params)]
        transitions,rewards = [jax.tree.map(lambda *args:jnp.stack(args,axis=0),*x) for x in zip(*res)]
    else:
        transitions,rewards = get_transitions(config, k4, paths, 20, ego_params=ego_params, mode="none", rewards_only=False, epsilon=0.01, track=True)
    
    # Train cluster ToM models
    print(f"Training Cluster ToM")
    ndata = get_batch_size(transitions)
    cluster_tom_uses_lstm = "cluster_tom" in config.get("USE_LSTM",{})
    cluster_tom_state,cluster_tom_losses = jax.vmap(partial(train,config,25,1000,use_lstm=cluster_tom_uses_lstm))(transitions, jax.random.split(k3,ndata))

    # Sample cluster BR + random cluster teammate trajectories
    print(f"Sampling inter-cluster trajectories")
    if ego_params is not None:
        res = [get_transitions(config,k4,p,1,ego_params=[tp]*len(p),mode="random",n_random=25,rewards_only=False,epsilon=0.01,track=True) for p in paths for tp in ego_params]
        transitions,rewards = [jax.tree.map(lambda *args:jnp.stack(args,axis=0),*x) for x in zip(*res)]
        transitions,rewards = jax.tree.map(lambda x:jnp.reshape(x,(len(paths),len(ego_params),*x.shape[1:])),(transitions,rewards))
    else:
        transitions,rewards = get_transitions(config,k1,sum(paths,start=[]),1,ego_params=ego_params,mode="cross_play",rewards_only=False,epsilon=0.1,track=True)
    
    # Train global ToM model
    print(f"Training Global ToM {paths}")
    global_tom_uses_lstm = "global_tom" in config.get("USE_LSTM",{})
    flat_transitions =  jax.tree.map(lambda x:x.reshape((-1,*x.shape[3:])), transitions)
    global_tom_state,global_tom_losses = train(config,25,2500,flat_transitions,k2,use_lstm=global_tom_uses_lstm)
    
    # Return models and avg final loss
    return global_tom_state,cluster_tom_state,global_tom_losses[-100:].mean(),cluster_tom_losses[:,-100:].mean(axis=-1)



def do_train_tom(config: dict):
    teammate_paths, ego_paths, _, _ = get_param_paths(config)
    
    # Train ToM
    global_tom_state,cluster_tom_state,global_tom_loss,cluster_tom_loss = train_tom_models(config, teammate_paths, ego_params=ego_paths)
    
    # Save
    alg_name = config.get("ALG_NAME", "tom_training")
    _, env_name = env_from_config(config)
    save_dir = os.path.join("overcooked_cache/tom_models", f'{alg_name}_{env_name}')

    os.makedirs(save_dir,exist_ok=True)
    save_params(global_tom_state.params, os.path.join(save_dir, "global_tom.safetensors"))
    save_params(cluster_tom_state.params, os.path.join(save_dir, "cluster_tom.safetensors"))
    with open(os.path.join(save_dir, "cfg.json"),"w") as f:
        json.dump(config,f)
    
    return global_tom_loss,cluster_tom_loss


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    config = {**config, **config["alg"]}

    env, _ = env_from_config(config)
    if set(config["EGO"]+config.get("TEAMMATE",[])) != set(range(env.num_agents)):
        raise ValueError("Agents must either be classified as EGO or TEAMMATE")
    if set(config["EGO"]).intersection(config["TEAMMATE"]) != set():
        raise ValueError("Agents cannot be classified as both EGO and TEAMMATE")

    print(do_train_tom(config))

if __name__ == "__main__":
    main()