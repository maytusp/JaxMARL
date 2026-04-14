import os
import jax
import jax.numpy as jnp
from functools import partial
from typing import Any

import optax
from flax.training.train_state import TrainState
import distrax
import json 
from tqdm import tqdm 

from utils.wrappers import (
    CTRolloutManager,
    load_params,
    save_params 
)
from env.communication import Communication
from models import MLPLSTM
from ctom import bce, communication_ctom, make_tom_target
from jax_tqdm import scan_tqdm


POP_ENV_KWARGS = {"num_classes":4,"reveal_correct":False,"reveal_class":True,"max_games":1}
ENV_KWARGS = {"num_classes":4,"reveal_correct":False,"reveal_class":True,"max_games":16}
POP_ENV= Communication(**POP_ENV_KWARGS) 
ENV  = Communication(**ENV_KWARGS) 
INFIX="reveal_class"
SEED=154

def env_name_from_kwargs(env_name,kwargs):
    n_classes = kwargs.get("num_classes",2)
    max_games = kwargs.get("max_games",1)
    if max_games in {64,32}:
        max_games=16
    env_name = f"{env_name}_{n_classes}classes_{max_games}games"
    if kwargs.get("reveal_class",False):
        env_name+="_reveal"
    if kwargs.get("force_action",False):
        env_name+="_force"
    return env_name 
    

def tree_batchify(x: dict):
    return jax.tree.map(lambda *args:jnp.stack(args,axis=0),*[x[agent] for agent in ["agent_0","agent_1"]])

def tree_unbatchify(x: jnp.ndarray):
    return {agent: jax.tree.map(lambda x_:x_[i],x) for i,agent in enumerate(["agent_0","agent_1"])}

def batchify(x: dict):
    return jnp.stack([x[agent] for agent in ["agent_0","agent_1"]])

def unbatchify(x: jnp.ndarray):
    return {agent: x[i] for i,agent in enumerate(["agent_0","agent_1"])}

def get_shape(x):
    return jax.tree.map(lambda x:x.shape,x)

def get_batch_size(x):
    return jax.tree.leaves(x)[0].shape[0]

def random_argmin(key,x,threshold=0.01):
    under_threshold = x<=(jnp.min(x)+threshold)
    logits = jnp.where(under_threshold,0,-10000) 
    pi = distrax.Categorical(logits=logits) 
    return pi.sample(seed=key)


class CriticTrainState(TrainState):
    batch_stats: Any 

def getdata(key):
    key,k1,k2,k3,k4 = jax.random.split(key,5)
    permk = jax.random.split(k1,2) 
    def get_single(k):
        k,k1,k2 = jax.random.split(k,3)
        m = jax.random.permutation(k1,4)
        inp = jax.random.randint(k2,(2,16),0,4) 
        otp = m[inp]
        return inp,otp 
    inp,otp = jax.vmap(get_single)(permk)
    inp = inp.reshape((-1,inp.shape[-1]))
    otp = otp.reshape((-1,otp.shape[-1]))
    return inp,otp 

def run_communication(env: Communication,map,key):
    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)
    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}
    wrapped_env = CTRolloutManager(
        env, batch_size=8, preprocess_obs=False
    )
    key,k1,k2,k3 = jax.random.split(key,4)
    obs,state = wrapped_env.batch_reset(k1)
    def step_fn(carry,unused):
        obs,env_state,key = carry 
        key,k1,k2,k3 = jax.random.split(key,4)

        action0 = jnp.where(env_state.current_agent==0,map[env_state.pet_id]+1,0)
        action1 = jax.random.choice(5,)
        # action1 = env_state.pet_id+1 
        actions = {"agent_0":action0,"agent_1":action1}
        print(f"shapes {get_shape(env_state)} {get_shape(actions)} {get_shape(k1)}")
        new_obs, new_env_state, reward, new_done, info = wrapped_env.batch_step(
            k1, env_state, actions
        )
        return (new_obs,new_env_state,key),(obs,env_state)
    _,(obs,env_state) = jax.lax.scan(step_fn,(obs,state,k2),length=ENV.max_games*2)
    inp = obs["agent_1"]
    otp = env_state.pet_id
    return inp,otp 



def run_communication_policy(env: Communication,num_envs,params,key):
    print(f"params shape {jax.tree.map(lambda x:x.shape,params)}")
    wrapped_env = CTRolloutManager(
        env, batch_size=num_envs, preprocess_obs=False
    )
    network = MLPLSTM(
        output_dim=wrapped_env.max_action_space,
        input_dims=(32,32),
        output_dims=(32,),
        lstm_dim=32,
        use_lstm=False
    )
    opp_network = network 

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)
    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}
    
    key,k1,k2,k3 = jax.random.split(key,4)
    obs,state = wrapped_env.batch_reset(k1)
    def step_fn(carry,unused):
        obs,hs,last_done,env_state,key = carry 
        key,k1,k2,k3 = jax.random.split(key,4)

        hs, pi = jax.vmap(network.apply, in_axes=(0,0, 0, None,None))(
            params,
            hs,
            batchify(obs)[:,None],  # (num_agents, num_envs, num_actions)
            last_done[None],
            False,
        )  # (num_agents, num_envs, num_actions)
        # explore
        avail_actions = wrapped_env.get_valid_actions(env_state)
        pi = distrax.Categorical(logits=pi.squeeze(1)-(1-batchify(avail_actions))*1000)

        actions = pi.sample(seed=k2)
        actions = unbatchify(actions)

        new_obs, new_env_state, reward, new_done, info = wrapped_env.batch_step(
            k1, env_state, actions
        )
        return (new_obs,hs,new_done["__all__"],new_env_state,key),(obs,env_state,reward)
    hs = MLPLSTM.initialize_carry(network.lstm_dim,env.num_agents,num_envs)
    _,(obs,env_state,reward) = jax.lax.scan(step_fn,(obs,hs,jnp.zeros(num_envs,dtype=jnp.bool_),state,k2),length=32)
    # jax.debug.print("reward {r}",r=jax.tree.map(lambda x:x.mean(axis=1),reward))
    inp = obs
    otp = env_state.pet_id
    return inp,otp 


def run_communication_single(config, env: Communication, params, teacher_params, key, tom_predictors=None, tom_observers=None, verbose=False):
    num_envs = 1
    wrapped_env = CTRolloutManager(
        env, batch_size=num_envs, preprocess_obs=False
    )
    network = MLPLSTM(
        output_dim=wrapped_env.max_action_space,
        input_dims=(32,32),
        output_dims=(32,),
        lstm_dim=32,
        use_lstm=False
    )

    br_network = MLPLSTM(
        output_dim=wrapped_env.max_action_space,
        input_dims=(32,32),
        output_dims=(32,),
        lstm_dim=32,
        use_lstm=True
    )
    if config.get("BR"):
        teacher_network = br_network
    else:
        teacher_network = network 
        # print(f"shapes {get_shape(params)}")
    
    tom_predictor = MLPLSTM(2*4,input_dims=(32,32),lstm_dim=32,output_dims=(32,),use_lstm=False,use_sigmoid=True,concept_shape=(2,4))
    tom_observer = MLPLSTM(2*4,input_dims=(32,32),lstm_dim=32,output_dims=(32,),use_lstm=True,use_sigmoid=True,concept_shape=(2,4))

    teacher_key = config.get("TEACHER","agent_1")
    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)
    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}
    
    method = config.get("method","manual")
    
    num_agents = get_batch_size(params)
    num_teachers = get_batch_size(teacher_params)
    if method in {"bootstrapped_concept","fine_tune"}:
        num_predictors = get_batch_size(tom_predictors) 
    
    key,k1,k2,k3,k4 = jax.random.split(key,5)

    random_agent_index = jax.random.choice(k3,num_agents) 
    params = jax.tree.map(lambda x:x[random_agent_index],params)

    if method=="manual":
        teacher_idx = random_agent_index 
    elif method in {"random once","bootstrapped_concept","fine_tune"}:
        teacher_idx = jax.random.choice(k4,num_teachers)
    else:
        raise ValueError(f"No such method {method}")
    
    steps_per_eval = config.get("STEPS_PER_EVAL",1)
    steps_to_keep = config.get("STEPS_TO_KEEP",1)
    initial_steps = config.get("INITIAL_STEPS",0)
    initial_teacher_idx = teacher_idx

    if verbose:
        jax.debug.print("\n ----- starting episode ---- \n")
        jax.debug.print("Initial opp {o} teacher {t}",o=random_agent_index,t=teacher_idx)

    def step_fn(carry,unused):
        timestep,obs,(hs,teacher_hs,predictor_hs,observer_hs),(saved_predictor,saved_observer, saved_avail),teacher_idx,last_done,env_state,key = carry 
        key,k1,k2,k3,k4,k5,k6 = jax.random.split(key,7)

        hs, pi = jax.vmap(network.apply, in_axes=(0,0, 0, None,None))(
            params,
            hs,
            batchify(obs)[:,None,None],  # (num_agents, num_envs, num_actions)
            last_done[None,None],
            False,
        )  # (num_agents, num_envs, num_actions)
        
        avail_actions = env.get_valid_actions(env_state)
        pi = distrax.Categorical(logits=pi.squeeze((1,2))-(1-batchify(avail_actions))*1000)
        actions = pi.sample(seed=k2)
        actions = unbatchify(actions)

        if verbose:
            jax.lax.cond(env_state.current_agent==0,lambda:jax.debug.print("\n",ordered=True),lambda:None)
            jax.debug.print("{time} method {m} opp {o} teacher {t} actions {a} game {g} agent {ag} pet_id {p}",o=random_agent_index,t=teacher_idx,m=method,a=actions,g=env_state.num_games,p=env_state.pet_id,ag=env_state.current_agent,time=timestep,ordered=True)

        if method in {"bootstrapped_concept","fine_tune"}:
            predictor_hs, predictor_output = jax.vmap(jax.vmap(tom_predictor.apply,in_axes=(0,0,0,None)),in_axes=(0,0,None,None))(tom_predictors,predictor_hs,batchify(obs)[:,None,None],last_done[None,None])
            predictor_output = jax.vmap(unbatchify)(predictor_output)[teacher_key].squeeze((1,2))
            observer_hs, observer_output = jax.vmap(tom_observer.apply,in_axes=(0,0,0,None))(tom_observers,observer_hs,batchify(obs)[:,None,None],last_done[None,None])
            observer_output = unbatchify(observer_output)[teacher_key].squeeze((0,1))

            saved_predictor = jnp.roll(saved_predictor,-1,axis=0).at[-1].set(predictor_output)
            saved_observer = jnp.roll(saved_observer,-1,axis=0).at[-1].set(observer_output)
            saved_avail = jnp.roll(saved_avail,-1,axis=0).at[-1].set(1)

            predictor_output,observer_output,avail = saved_predictor[-steps_to_keep:],saved_observer[-steps_to_keep:],saved_avail[-steps_to_keep:]
            predictor_losses = jax.vmap(partial(bce,reduction="none"),in_axes=(1,None))(predictor_output,observer_output)
            predictor_losses = predictor_losses.mean(axis=range(2,predictor_losses.ndim)) 
            total_losses = (predictor_losses*avail).sum(axis=-1)/avail.sum() 

            new_teacher_idx = random_argmin(k4,total_losses)
            if verbose:
                jax.debug.print("{time} predictor output \n{p}\n{p_} \nobserver output \n{o}",p=predictor_output.round(2).argmax(axis=-1)[...,0],p_=predictor_output.round(2).max(axis=-1)[...,0],o=observer_output.round(2)[...,0,:],time=timestep,ordered=True)
                jax.debug.print("{time} {m} prev {p} opp {o} total losses {t} chosen {c}",m=method,p=teacher_idx, t=total_losses, o=random_agent_index,c=new_teacher_idx,time=timestep,ordered=True)
        else:
            new_teacher_idx = teacher_idx

        reselect_teacher = (timestep>=initial_steps) & ((timestep-initial_steps)%steps_per_eval)==0
        teacher_idx = jnp.where(reselect_teacher,new_teacher_idx,teacher_idx)

        teacher_hs, teacher_pi = jax.vmap(jax.vmap(teacher_network.apply, in_axes=(0,0, 0, None,None)),in_axes=(0,0,None,None,None))(
            teacher_params,
            teacher_hs,
            batchify(obs)[:,None,None],  # (num_agents, num_envs, num_actions)
            last_done[None,None],
            False,
        )  # (num_agents, num_envs, num_actions)

        # if method in {"bootstrapped_concept","fine_tune"}:
        #     teacher_pi = teacher_pi.at[...,1:].set(jnp.log(observer_output[-1,0]*0.999+0.005))
        #     teacher_pi = teacher_pi.at[...,0].set(-100)
        # teacher_pi = jnp.where(timestep>=initial_steps,teacher_pi,teacher_pi.at[...,1:].set(100))
        teacher_pi = distrax.Categorical(logits=teacher_pi.squeeze((2,3))-(1-batchify(avail_actions)[None])*1000)
        teacher_actions = teacher_pi.sample(seed=k2)
        teacher_actions = jax.vmap(unbatchify)(teacher_actions) 
        actions[teacher_key]=teacher_actions[teacher_key][teacher_idx]
        actions[teacher_key] = jnp.where(env_state.current_agent==1,actions[teacher_key],0)
        
        new_obs, new_env_state, reward, new_done, info = env.step(k1, env_state, actions)

        if verbose:
            jax.debug.print("{time} teacher action {t} reward {r}",t=teacher_actions[teacher_key][teacher_idx],time=timestep,r=reward[teacher_key],ordered=True)

        return (timestep+1,new_obs,(hs,teacher_hs,predictor_hs,observer_hs),(saved_predictor,saved_observer, saved_avail),teacher_idx,new_done["__all__"],new_env_state,key),(obs,env_state,reward,actions)
    
    num_steps = ENV.max_games*2

    obs,state = env.reset(k1)
    
    hs = MLPLSTM.initialize_carry(network.lstm_dim,env.num_agents,num_envs)
    teacher_hs = MLPLSTM.initialize_carry(network.lstm_dim,num_teachers,env.num_agents,num_envs)
    if method in {"bootstrapped_concept","fine_tune"}:
        predictor_hs = MLPLSTM.initialize_carry(tom_predictor.lstm_dim,num_predictors,env.num_agents,num_envs)
        observer_hs = MLPLSTM.initialize_carry(tom_observer.lstm_dim,env.num_agents,num_envs)
    else:
        predictor_hs=observer_hs=None
    init_hs = (hs,teacher_hs,predictor_hs,observer_hs) 

    if method in {"bootstrapped_concept", "fine_tune"}:
        saved_predictor = jnp.zeros((num_steps,num_predictors,env.num_agents,env.num_classes)) 
        saved_observer = jnp.zeros((num_steps,env.num_agents,env.num_classes)) 
        saved_avail = jnp.zeros(num_steps)
        prev_loss = (saved_predictor,saved_observer,saved_avail)
    else:
        prev_loss = (None,None,None)

    _,(obs,env_state,reward,actions) = jax.lax.scan(step_fn,(1,obs,init_hs,prev_loss,teacher_idx,jnp.bool_(False),state,k2),length=num_steps)
    
    
    total_reward = reward[teacher_key].sum()
    actions = (actions[teacher_key]==0).sum()
    # jax.debug.print("random agent {a} initial teacher {t} reward {r}",r=total_reward,a=random_agent_index,t=initial_teacher_idx)
    return {"total_returns":total_reward,"bailout_frequency":actions}

def run_communication_multiple(config,env,params,teacher_params,key,tom_predictors=None,tom_observers=None,n_iter=1,verbose=False):
    method = config.get("method","manual")
    if method=="fine_tune":
        key,k1 = jax.random.split(key)
        fine_tune_params = config.get("fine_tune_params",{})
        tom_observers = fine_tune_observer(tom_observers,params,k1,**fine_tune_params) 
    def step_fn(key,unused):
        key,k1 = jax.random.split(key)
        res = run_communication_single(config,env,params,teacher_params,k1,tom_predictors=tom_predictors,tom_observers=tom_observers,verbose=verbose) 
        return key,res 
    _,res = jax.lax.scan(step_fn,key,length=n_iter) 
    res = jax.tree.map(lambda x:x.tolist(),res)
    return res        

def get_communication_data(key):
    key,k1,k2,k3,k4 = jax.random.split(key,5)
    permk = jax.random.split(k1,1000)  
    def get_single(k):
        k,k1,k2 = jax.random.split(k,3)
        m = jax.random.permutation(k1,4) # map 
        inp,otp = run_communication(ENV,m,k2)
        return inp,otp 
    inp,otp = jax.vmap(get_single)(permk)
    inp = inp.swapaxes(1,2)
    otp = otp.swapaxes(1,2)
    print(f"inp and otp {get_shape(inp)} {get_shape(otp)}")
    inp = inp.reshape((-1,*inp.shape[-2:]))
    otp = otp.reshape((-1,otp.shape[-1]))
    return inp,otp 

def get_communication_policy_data(key,n_policies,num_envs,cross_play=False, params_override=None, random=False, n_random=0):
    key,k1,k2,k3,k4 = jax.random.split(key,5)

    prefix = env_name_from_kwargs("communication",POP_ENV_KWARGS)
    if params_override is None:
        paths = [f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_teacher_{prefix}/seed{SEED}_vmap{i}.safetensors" for i in range(n_policies)]
        params = [tree_batchify(load_params(p)) for p in paths]
        params = jax.tree.map(lambda *args:jnp.stack(args),*params)
    else:
        params = params_override
        n_policies = get_batch_size(params)

    permk = jax.random.split(k1,n_policies) 
    if cross_play:
        def make_xplay(x):
            i,j = jnp.meshgrid(jnp.arange(x.shape[0]),jnp.arange(x.shape[0]),indexing="ij") 
            x = jnp.stack([x[i,0],x[j,1]],axis=2) 
            return x.reshape([-1,*x.shape[2:]])
        params = jax.tree.map(make_xplay,params)
        permk = jax.vmap(lambda x:jax.random.split(x,n_policies))(permk).reshape(-1)
    if random:
        i, j = jax.random.choice(k2,n_policies,(n_random*2,)).reshape([2,-1]) 
        def getij(i,j):
            i = jax.tree.map(lambda x:x[i,0],params) 
            j = jax.tree.map(lambda x:x[j,1],params) 
            return tree_batchify({"agent_0":i,"agent_1":j}) 
        params = jax.vmap(getij)(i,j) 
        permk = jax.random.split(k1,n_random) 
    
    def get_single(k,p):
        k,k1,k2 = jax.random.split(k,3)
        inp,otp = run_communication_policy(ENV,num_envs,p,k2)
        return inp,otp 
    inp,otp = jax.vmap(get_single)(permk,params)

    print(f"inp otp {get_shape(inp)} {get_shape(otp)}")
    inp = jax.tree.map(lambda x:x.swapaxes(1,2),inp)
    otp = otp.swapaxes(1,2)

    if cross_play:
        inp,otp = jax.tree.map(lambda x:x.reshape([n_policies,n_policies,*x.shape[1:]]),(inp,otp))

    return inp,otp 

def get_communication_eval_data(key,n_policies,n_eval_policies=25):
    key,k1,k2,k3,k4 = jax.random.split(key,5)

    prefix = env_name_from_kwargs("communication",POP_ENV_KWARGS)
    paths = [f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_teacher_{prefix}/seed{SEED}_vmap{i}.safetensors" for i in range(n_policies)]
    ood_paths = [f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_teacher_{prefix}/seed{SEED}_vmap{i}.safetensors" for i in range(25,25+n_eval_policies)]
    params = [tree_batchify(load_params(p)) for p in paths]
    params = jax.tree.map(lambda *args:jnp.stack(args),*params)
    
    ood_params = [tree_batchify(load_params(p)) for p in ood_paths]
    ood_params = jax.tree.map(lambda *args:jnp.stack(args),*ood_params)

    prefix = env_name_from_kwargs("communication",ENV_KWARGS)
    tom_observers = load_params(f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_predictor{n_policies}_{prefix}/seed{SEED}_vmap_obs.safetensors")
    best_observer = load_params(f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_predictor{25}_{prefix}/seed{SEED}_vmap_obs.safetensors")
    tom_predictors = load_params(f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_predictor{n_policies}_{prefix}/seed{SEED}_vmap_pred.safetensors")

    br_paths = [f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_br2_{prefix}/seed{SEED}_vmap{25*i+n_policies-1}.safetensors" for i in range(10)]
    br_params = [tree_batchify(load_params(p)) for p in br_paths]
    br_params = jax.tree.map(lambda *args:jnp.stack(args),*br_params)
    # br_params = None

    generalist_paths = [f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_static_br_{prefix}/seed{SEED}_vmap{25*i+n_policies-1}.safetensors" for i in range(10)]
    generalist_params = [tree_batchify(load_params(p)) for p in generalist_paths]
    generalist_params = jax.tree.map(lambda *args:jnp.stack(args),*generalist_params)

    fine_tune_ratio = 1
    config = {"TEACHER":"agent_1",
              "STEPS_PER_EVAL":1,
              "STEPS_TO_KEEP":1,
              "INITIAL_STEPS":4,
              "method":"manual",
              "fine_tune_params":{
                  "n_random":max(1,int(n_policies/fine_tune_ratio)),
                  "n_envs":1,
                  "lr":0.0001,
                  "total_iter":1000}}
    
    n_iters = 100
    # in distribution 
    manual = run_communication_multiple({**config,**{"method":"manual"}}, ENV, params, params, k2, n_iter= n_iters)
    random = run_communication_multiple({**config,**{"method":"random once"}}, ENV, params, params, k2, n_iter= n_iters)
    concept = run_communication_multiple({**config,**{"method":"bootstrapped_concept"}}, ENV, params, params, k2, tom_predictors=tom_predictors,tom_observers=tom_observers,n_iter= n_iters,verbose=False)
    # concept_best = run_communication_multiple({**config,**{"method":"bootstrapped_concept"}}, ENV, params, params, k2, tom_predictors=tom_predictors,tom_observers=best_observer,n_iter= n_iters,verbose=False)
    # concept_fine_tune = run_communication_multiple({**config,**{"method":"fine_tune"}}, ENV, params, params, k2, tom_predictors=tom_predictors,tom_observers=tom_observers,n_iter= n_iters,verbose=False)
    # concept_best_fine_tune = run_communication_multiple({**config,**{"method":"fine_tune"}}, ENV, params, params, k2, tom_predictors=tom_predictors,tom_observers=best_observer,n_iter= n_iters,verbose=False)
    br = run_communication_multiple({**config,**{"method":"random once","BR":True}}, ENV, params, br_params, k2, n_iter= n_iters)
    # generalist = run_communication_multiple({**config,**{"method":"random once","BR":False}}, ENV, params, generalist_params, k2, n_iter= n_iters)
    in_dist = {
        "manual":manual,
        "random":random,
        "concept":concept,
        # "concept_best":concept,
        # "concept_fine_tune":concept_fine_tune,
        # "concept_best_fine_tune":concept_best_fine_tune,
        "br":br,
        # "generalist":generalist,
               }
    # in_dist=None
    # print(f"in dist reward concept {concept['total_returns']}")#\nbr {br['total_returns']}")
    # print(f"avg in dist rewards {jnp.mean(jnp.array(concept['total_returns']))}  {jnp.mean(jnp.array(concept['total_returns']))}")
    # print(f"avg in dist rewards {jnp.mean(jnp.array(br['total_returns']))}  {jnp.mean(jnp.array(br['total_returns']))}")
    # exit(0)
    # # out of distribution
    manual = run_communication_multiple({**config,**{"method":"manual"}}, ENV, ood_params, ood_params, k2, n_iter= n_iters)
    random = run_communication_multiple({**config,**{"method":"random once"}}, ENV, ood_params, params, k2, n_iter= n_iters)
    concept = run_communication_multiple({**config,**{"method":"bootstrapped_concept"}}, ENV, ood_params, params, k2, tom_predictors=tom_predictors,tom_observers=tom_observers,n_iter= n_iters)
    # concept_best = run_communication_multiple({**config,**{"method":"bootstrapped_concept"}}, ENV, ood_params, params, k2, tom_predictors=tom_predictors,tom_observers=best_observer,n_iter= n_iters,verbose=False)
    # concept_fine_tune = run_communication_multiple({**config,**{"method":"fine_tune"}}, ENV, ood_params, params, k2, tom_predictors=tom_predictors,tom_observers=tom_observers,n_iter= n_iters,verbose=False)
    # concept_best_fine_tune = run_communication_multiple({**config,**{"method":"fine_tune"}}, ENV, ood_params, params, k2, tom_predictors=tom_predictors,tom_observers=best_observer,n_iter= n_iters,verbose=False)
    br = run_communication_multiple({**config,**{"method":"random once","BR":True}}, ENV, ood_params, br_params, k2, n_iter= n_iters)
    # generalist = run_communication_multiple({**config,**{"method":"random once","BR":False}}, ENV, ood_params, generalist_params, k2, n_iter= n_iters)
    ood = {
        "manual":manual,
        "random":random,
        "concept":concept,
        # "concept_best":concept,
        # "concept_fine_tune":concept_fine_tune,
        # "concept_best_fine_tune":concept_best_fine_tune,
        "br":br
        # "generalist":generalist,
           }
    # ood = None
    # print(f"avg out of dist rewards {jnp.mean(jnp.array(concept['total_returns']))}  {jnp.mean(jnp.array(concept['total_returns']))}")
    
    return {"in_dist":in_dist,"ood":ood}

def train(n_epochs,n_steps,inp,otp,key,tom_output=None,use_lstm=True,init_params=None,lr_init=None, perm_invariant=False):
    key,k1,k2,k3,k4 = jax.random.split(key,5)
    model = MLPLSTM(2*4,input_dims=(32,32),lstm_dim=32,output_dims=(32,),use_lstm=use_lstm,use_sigmoid=True,concept_shape=(2,4))
    
    print(f"data shapes {get_shape(inp)} {get_shape(otp)} {get_shape(tom_output)}")
    ndata = otp.shape[0]
    obs_dim = batchify(inp).shape[-1]

    def create_state(key):
        init_x = jnp.zeros((1,1,obs_dim))
        init_done = jnp.zeros((1,1))
        hidden = MLPLSTM.initialize_carry(model.lstm_dim,1) 
        variables = model.init(key,hidden,init_x,init_done)
        lr_scheduler = optax.cosine_decay_schedule(lr_init or 0.001,n_steps*n_epochs)
        tx = optax.chain(
            optax.clip_by_global_norm(10),
            optax.radam(learning_rate=lr_scheduler)
        )
        train_state = CriticTrainState.create(apply_fn=model.apply,params=variables["params"],batch_stats=variables["batch_stats"],tx=tx)
        return train_state 
    train_state = jax.vmap(create_state)(jax.random.split(k1,2))
    if init_params is not None:
        train_state = train_state.replace(params=init_params["params"],batch_stats=init_params["batch_stats"])
    
    def get_batch(key,bsz=16):
        data = jax.random.choice(key,ndata,(bsz,))
        if tom_output is not None:
            return jax.tree.map(lambda x:x[data],(inp,otp,tom_output))
        else:
            return jax.tree.map(lambda x:x[data],inp),otp[data]
    
    def train_iter(carry,unused,bsz=16):
        train_state,key = carry
        key,k1,k2 = jax.random.split(key,3)
        if tom_output is not None:
            inp,otp,tom = get_batch(k1,bsz=bsz)
            print(f"train shapes {get_shape(inp)} {get_shape(otp)} {get_shape(tom)}")
        else:
            inp,otp = get_batch(k1,bsz=bsz)
            print(f"train shapes {get_shape(inp)} {get_shape(otp)}")
        
        dones = jnp.zeros(otp.shape)
        if perm_invariant:
            perm = jax.random.permutation(k2,4) 
            otp = perm[otp]

        otp = jax.nn.one_hot(otp,4).swapaxes(0,1)
        inp = jax.tree.map(lambda x:x.swapaxes(0,1),inp)
        if tom_output is not None:
            tom = jax.tree.map(lambda x:x.swapaxes(0,1),tom)
        dones = dones.swapaxes(0,1)

        def loss_fn(params):
            hidden = MLPLSTM.initialize_carry(model.lstm_dim,2,bsz) 
            print(f"pre apply {get_shape(hidden)} {get_shape(inp)} {get_shape(dones)}")
            _,out = jax.vmap(model.apply,in_axes=(0,0,0,None))({"params":params,"batch_stats":train_state.batch_stats},hidden,batchify(inp),dones)
            
            out = jnp.moveaxis(out,0,2)
            print(f"shapes {get_shape(out)} {get_shape(otp)}")

            make_all_targets = lambda output,gt:jax.vmap(make_tom_target,in_axes=(None,0,0))(output,jnp.stack([gt,gt]),jnp.arange(2))
            tom_out = out if tom_output is None else tom
            tom_out = out
            tom_targets = jax.vmap(jax.vmap(lambda o,t:make_all_targets(o,t)))(tom_out,otp) # (num_steps, num_envs, num_agents, num_agents, concept_dim)
            
            tom_loss = bce(out*0.999+0.0005,jax.lax.stop_gradient(tom_targets)*0.999+0.0005)

            return tom_loss,out
        (loss,out),grads = jax.value_and_grad(loss_fn,has_aux=True)(train_state.params) 
        train_state = jax.vmap(lambda t,g:t.apply_gradients(grads=g))(train_state,grads)

        # print(f"inp {jax.tree.map(lambda x:x.swapaxes(0,1)[0],inp)['agent_1']}")
        # print(f"out {out.swapaxes(0,1)[0].argmax(axis=-1)[...,1,0]} \n {out.swapaxes(0,1)[0].max(axis=-1)[...,1,0]}")
        # print(f"otp {otp.swapaxes(0,1)[0].argmax(axis=-1)}")
        # if tom_output is not None:
        #     print(f"tom {jax.tree.map(lambda x:x.swapaxes(0,1)[0],tom)}")

        return (train_state,key),loss 
    
    bsz = min(16,ndata)
    jit_train_iter = jax.jit(scan_tqdm(n_steps)(partial(train_iter,bsz=bsz)))
    for i in tqdm(range(n_epochs)):
        print("\n\n")
        (train_state,_),losses = jax.lax.scan(jit_train_iter,(train_state,k3),xs=jnp.arange(n_steps),length=n_steps) 
        print(losses.reshape([25,-1]).mean(axis=-1).round(3))
    
    train_iter((train_state,k3),None,bsz=16)
    return train_state,losses

def get_single_tom_target(inp,predictor_params):
    tom_predictor = MLPLSTM(2*4,input_dims=(32,32),lstm_dim=32,output_dims=(32,),use_lstm=False,use_sigmoid=True,concept_shape=(2,4))
    inp = jax.tree.map(lambda x:x.swapaxes(0,1),inp)
    bsz = get_batch_size(inp)
    hidden = MLPLSTM.initialize_carry(tom_predictor.lstm_dim,2,bsz) 
    dones = jnp.zeros(jax.tree.leaves(inp)[0].shape[:2])
    _,out = jax.vmap(tom_predictor.apply,in_axes=(0,0,0,None))(predictor_params,hidden,batchify(inp),dones)
    out = jnp.moveaxis(out,0,2)
    out = jnp.swapaxes(out,0,1)
    return out 

def get_tom_target(inp,predictor_state):
    ndata = get_batch_size(inp)
    params = {"params":predictor_state.params,"batch_stats":predictor_state.batch_stats}
    def make_xplay(x):
        i,j = jnp.meshgrid(jnp.arange(x.shape[0]),jnp.arange(x.shape[0]),indexing="ij") 
        x = jnp.stack([x[i,0],x[j,1]],axis=2) 
        return x
    params = jax.tree.map(make_xplay,params)
    otp = jax.vmap(jax.vmap(get_single_tom_target))(inp,params) 
    return otp 

def train_all(key,n_policies):
    total_n_envs = 30000
    key,k1,k2,k3 = jax.random.split(key,4) 
    inp,otp = get_communication_policy_data(k1,n_policies,1000)
    ndata = get_batch_size(inp)
    print(f"n policies {ndata}")

    predictor_state,predictor_losses = jax.vmap(partial(train,5,1000,use_lstm=False))(inp,otp,jax.random.split(k3,ndata))

    obs_inp,obs_otp = get_communication_policy_data(k1,n_policies,total_n_envs//(n_policies*n_policies),cross_play=True)
    print(f"obs_inp {get_shape(obs_inp)} obs_otp {get_shape(obs_otp)}")
    tom_output = get_tom_target(obs_inp,predictor_state)
    print(f"tom output {get_shape(tom_output)}")
    
    flat_inp,flat_otp,flat_target =  jax.tree.map(lambda x:x.reshape((-1,*x.shape[3:])),(obs_inp,obs_otp,tom_output))
    observer_state,observer_losses = train(15,5000,flat_inp,flat_otp,k2,tom_output=flat_target,perm_invariant=False)
    
    return observer_state,predictor_state,observer_losses[-100:].mean(),predictor_losses[:,-100:].mean(axis=-1)

def fine_tune_observer(observer,policy_params,key,n_random=1,n_envs=1,lr=0.0001,total_iter=5000):
    key,k1,k2,k3 = jax.random.split(key,4) 
    obs_inp,obs_otp = get_communication_policy_data(k1,1,n_envs,params_override=policy_params,random=True,n_random=n_random)

    flat_inp,flat_otp =  jax.tree.map(lambda x:x.reshape((-1,*x.shape[2:])),(obs_inp,obs_otp))
    observer_state,observer_losses = train(1,total_iter,flat_inp,flat_otp,k2,lr_init=lr,init_params=observer)

    return {"params":observer_state.params,"batch_stats":observer_state.batch_stats}
    
def do_train_observer(n):
    observer_state,predictor_state,observer_losses,predictor_losses = train_all(jax.random.key(12),n)
    prefix = env_name_from_kwargs("communication",ENV_KWARGS)
    os.makedirs(f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_predictor{n}_{prefix}",exist_ok=True)
    save_params({"params":observer_state.params,"batch_stats":observer_state.batch_stats},f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_predictor{n}_{prefix}/seed{SEED}_vmap_obs.safetensors")
    save_params({"params":predictor_state.params,"batch_stats":predictor_state.batch_stats},f"/home/acni/pworld/save/communication/{prefix}/br_adaptation-{INFIX}-train_predictor{n}_{prefix}/seed{SEED}_vmap_pred.safetensors")
    return observer_losses,predictor_losses

def do_communication_eval(n,n_eval=25):
    communication_eval_data = get_communication_eval_data(jax.random.key(12),n,n_eval_policies=n_eval) 
    os.makedirs(f"/home/acni/pworld/communication_cache/{INFIX}-ctom_eval-bailout",exist_ok=True)
    with open(f"/home/acni/pworld/communication_cache/{INFIX}-ctom_eval-bailout/{n}_policies-{n_eval}_eval.json","w") as f:
        json.dump(communication_eval_data,f)

obs_loss = []
pred_loss = []
for i in tqdm(range(1,26)):
    # for j in tqdm(range(1,26)):
    for j in [i]:
        # print(f"\n{(i,j)}\n")
        # o,p = do_train_observer(i)
        # obs_loss.append(o)
        # pred_loss.append(p)
        do_communication_eval(i,n_eval=j)
    
print(obs_loss)
print(pred_loss)
