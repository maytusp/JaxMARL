import os
import copy
import jax
import jax.numpy as jnp
from functools import partial
from typing import Any

import chex
import optax
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import wandb
import distrax
import json 

from utils.wrappers import (
    LogWrapper,
    CTRolloutManager,
    load_params,
    save_params 
)
from env.communication import Communication
from models import MLP,MLPLSTM
from ctom import bce 
from jax_tqdm import scan_tqdm

def get_shape(x):
    return jax.tree.map(lambda x:x.shape,x)

def scan_loop(f,s=1):
    def wrapper(*arg):
        arg = jax.tree.map(lambda x:jnp.reshape(x,(-1,s,*x.shape[1:])),arg)
        _,res = jax.lax.scan(lambda a,b:(a,jax.vmap(f)(*b)),0,xs=arg)
        res = jax.tree.map(lambda x:jnp.reshape(x,(-1,*x.shape[2:])),res)
        return res 
    return wrapper

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
    logp: Any = None 

class CriticTrainState(TrainState):
    batch_stats: Any 

class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    remaining_advice: int = 0
    club_train_state: Any = None
    teacher_params: Any = None 
    teacher_opt_state: TrainState = None 
    opponent_params: Any = None 
    predictor_params: Any = None 
    predictor_train_state: Any = None 
    critic_train_state: Any = None 


def make_train(config, env):
    def tree_batchify(x: dict):
        return jax.tree.map(lambda *args:jnp.stack(args,axis=0),*[x[agent] for agent in env.agents])
    def broadcast_back_to(x,y):
        missing_dims = max(0,y.ndim-x.ndim)
        return jnp.reshape(x,x.shape + (1,)*missing_dims)

    assert (
        (config["NUM_ENVS"]*config["NUM_STEPS"]) % config["NUM_MINIBATCHES"] == 0
    ), "NUM_ENVS*NUM_STEPS must be divisible by NUM_MINIBATCHES"

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * config["NUM_UPDATES"],
    )
    
    rew_shaping_horizon = config["REW_SHAPING_HORIZON"]
    if rew_shaping_horizon>0:
        rew_shaping_horizon = config["TOTAL_TIMESTEPS"]*rew_shaping_horizon if rew_shaping_horizon<=1 else rew_shaping_horizon
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.0, end_value=0.0, transition_steps=rew_shaping_horizon
        )
    else:
        rew_shaping_anneal = lambda *args:0.

    ent_horizon = config["ENT_HORIZON"]
    if ent_horizon>0:
        ent_horizon = config["TOTAL_TIMESTEPS"]*ent_horizon if ent_horizon<=1 else ent_horizon
        ent_anneal = optax.linear_schedule(
            init_value=config["ENT_COEF"], end_value=config.get("ENT_FINISH",0), transition_steps=ent_horizon
        )
    else:
        ent_anneal = lambda *args:config["ENT_COEF"]

    alpha_horizon = config.get("MEP_ALPHA_HORIZON",0)
    if alpha_horizon>0:
        alpha_horizon = config["TOTAL_TIMESTEPS"]*alpha_horizon if alpha_horizon<=1 else alpha_horizon
        alpha_anneal = optax.linear_schedule(
            init_value=config["MEP_ALPHA"], end_value=0.0, transition_steps=alpha_horizon
        )
    else:
        alpha_anneal = lambda *args:config.get("MEP_ALPHA",0)

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    # epsilon-greedy exploration
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
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            random_actions,
            greedy_actions,
        )
        return chosed_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(input):
        rng,seed_id,init_params = input 
        jax.debug.print("seed id {s}",s=seed_id)
        NUM_TEACHERS = config.get("NUM TEACHERS",0)

        original_seed = rng[0]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(
            env, batch_size=config["NUM_ENVS"], preprocess_obs=False
        )
        test_env = CTRolloutManager(
            env,
            batch_size=config["TEST_NUM_ENVS"],
            preprocess_obs=False,
        )

        ctom_version = config.get("CTOM_VERSION",1)
        action_anticipation = config.get("ACTION_ANTICIPATION",0)


        # INIT NETWORK AND OPTIMIZER
        network = MLPLSTM(
            output_dim=wrapped_env.max_action_space,
            input_dims=(32,32),
            output_dims=(32,),
            lstm_dim=32,
            use_lstm=False
        )
        opp_network = network 

        
        br_network = MLPLSTM(
            output_dim=wrapped_env.max_action_space,
            input_dims=(32,32),
            output_dims=(32,),
            lstm_dim=32,
            use_lstm=True
        )

        if config.get("BR"):
            network = br_network

        critic = MLPLSTM(
            output_dim=1,
            input_dims=(32,32),
            output_dims=(32,),
            lstm_dim=32,
            use_lstm=False,
            critic = True
        )

        br_critic = MLPLSTM(
            output_dim=1,
            input_dims=(32,32),
            output_dims=(32,),
            lstm_dim=32,
            use_lstm=True,
            critic = True
        )

        if config.get("BR"):
            critic = br_critic

        def create_agent(rng):
            rng,rng_,rng_pred = jax.random.split(rng,3)
            init_x = jnp.zeros((1, *env.observation_space().shape))
            init_done = jnp.zeros((1, 1))
            init_carry = MLPLSTM.initialize_carry(network.lstm_dim, 1)
            network_variables = network.init(rng_, init_carry, init_x, init_done, train=False)
            init_x = jnp.zeros((1, env.observation_space().shape[-1]*env.num_agents))
            critic_variables = critic.init(rng_, init_carry, init_x, init_done, train=False)
            

            lr_scheduler = optax.linear_schedule(
                config["LR"],
                config["LR"]*config.get("LR_DECAY_FINISH",0),
                config["NUM_EPOCHS"]
                * config["NUM_MINIBATCHES"]
                * config["NUM_UPDATES"]
                * config.get("LR_DECAY_TIME",1),
            )
            
            exp_scheduler = optax.exponential_decay(
                config["LR"],
                config["NUM_EPOCHS"]
                * config["NUM_MINIBATCHES"]
                * config["NUM_UPDATES"],
                config.get("LR_DECAY_FINISH",0.01)
            )
            lr = {"LINEAR":lr_scheduler,"EXP":exp_scheduler,"NONE":config["LR"]}[config.get("LR_DECAY","NONE")]
            
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr)
            )

            critic_train_state = CriticTrainState.create(
                apply_fn=critic.apply,
                params=critic_variables["params"],
                batch_stats=critic_variables["batch_stats"],
                tx=tx)

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx
            )

            return train_state,critic_train_state
        
        rng,_rng = jax.random.split(rng)

        train_state,critic_train_state = jax.vmap(create_agent)(jax.random.split(rng,env.num_agents))
        critic_train_state = jax.tree.map(lambda x:x[0],critic_train_state)
        if init_params is not None:
            train_state = train_state.replace(params=init_params)
    
        ### Get teacher and opponent params ###
        teacher_params = config.get("TEACHER PARAMS",None)
        opponent_params = config.get("OPPONENT PARAMS",None)
        teacher_params = jax.tree.map(lambda x:jnp.swapaxes(x,0,1),teacher_params)
        opponent_params = jax.tree.map(lambda x:jnp.swapaxes(x,0,1),opponent_params)
        train_state = train_state.replace(teacher_params=teacher_params,opponent_params=opponent_params)
        if teacher_params is not None and config.get("FINE TUNE TEACHER CONCEPTS", False):
            opt_state = TrainState.create(
                apply_fn = train_state.apply_fn, 
                params = teacher_params,
                tx = train_state.tx,)
            train_state = train_state.replace(teacher_opt_state = opt_state)

        #### Opponent and teacher selection
        OPPONENT_IDX=None
        if opponent_params is not None:
            if config.get("OPPONENT SELECTION","random once")=="random once":
                rng,_rng = jax.random.split(rng)
                opponent_permutation = jax.random.permutation(_rng,config.get("NUM OPPONENTS",1))
                # opponent_permutation = jnp.array([0,1,2,3,4])
                OPPONENT_IDX = opponent_permutation[0]
                # Remove corresponding teacher parameter
                if config.get("TEACHER SELECTION","random") in {"majority vote","play_episode_predictor"}:
                    NUM_TEACHERS = NUM_TEACHERS-1
                    new_params = jax.tree.map(lambda x:x[:,opponent_permutation[1:]],train_state.teacher_params)
                    train_state = train_state.replace(teacher_params=new_params)
                    if teacher_params is not None and config.get("FINE TUNE TEACHER CONCEPTS", False):
                        opt_state = TrainState.create(
                            apply_fn = train_state.apply_fn, 
                            params = new_params,
                            tx = train_state.tx,)
                        train_state = train_state.replace(teacher_opt_state = opt_state)
            else:
                OPPONENT_IDX=0
        ####################################

        rng, _rng = jax.random.split(rng)
        

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_states, save_state, expl_state, test_state, remaining_advice, _, rng = runner_state
            train_state,critic_train_state = train_states
            teacher_frequencies = jnp.zeros((max(1,NUM_TEACHERS),))
            teacher_losses = jnp.zeros((max(1,NUM_TEACHERS),))

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs,last_done, hiddens, (opp_ids,heuristic_ids), env_state, rem_advice, teacher_frequencies, teacher_losses, rng = carry
                hs,critic_hs,opp_hs,pop_hs=hiddens
                
                rng, rng_a, rng_s,rng_pi = jax.random.split(rng, 4)
                _obs = batchify(last_obs)
                hs, pi = jax.vmap(network.apply, in_axes=(0,0, 0, None,None))(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs[:,None],  # (num_agents, num_envs, num_actions)
                    last_done[None],
                    False,
                )  # (num_agents, num_envs, num_actions)
                # explore
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
                pi = distrax.Categorical(logits=pi.squeeze(1)-(1-batchify(avail_actions))*1000)

                actions = pi.sample(seed=rng_pi)
                log_prob = pi.log_prob(actions)
                actions = unbatchify(actions)
                log_prob = unbatchify(log_prob)

                critic_hs, q_vals = critic.apply(
                    {
                        "params": critic_train_state.params,
                        "batch_stats": critic_train_state.batch_stats,
                    },
                    critic_hs,
                    last_obs["__all__"][None],  # (num_agents, num_envs, num_actions)
                    last_done[None],
                    False,
                )  # (num_agents, num_envs, num_actions)
                q_vals = q_vals.squeeze(0)

                ###### Opponent and Teacher Actions #####
                if config.get("OPPONENT PARAMS",None) is not None:
                    opp_params = jax.tree.map(lambda x:jnp.swapaxes(x,0,1),train_state.opponent_params)
                    # print(f"shapes {get_shape(train_state.opponent_params)} {get_shape(opp_hs)} {get_shape(_obs)}")
                    opp_hs, opp_pi = jax.vmap(jax.vmap(opp_network.apply, in_axes=(0,0, 0, None,None)),in_axes=(0,None,None,None,None))(
                        opp_params,
                        opp_hs,
                        _obs[:,None],  # (num_agents, num_envs, num_actions)
                        last_done[None],
                        False,
                    )  # (num_policies, num_agents, num_envs, num_actions)
                    # explore
                    opp_pi = opp_pi.squeeze(2)
                    opp_hs,opp_pi = jax.tree.map(lambda x:x[opp_ids,:,jnp.arange(config["NUM_ENVS"])].swapaxes(0,1),(opp_hs,opp_pi)) # num_envs, num_agents
                    avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
                    opp_pi = distrax.Categorical(logits=opp_pi-(1-batchify(avail_actions))*1000)

                    opp_actions = opp_pi.sample(seed=rng_pi)
                    opp_log_prob = pi.log_prob(opp_actions)
                    opp_actions = unbatchify(opp_actions)
                    opp_log_prob = unbatchify(opp_log_prob)

                    print(f"shapes {get_shape(opp_actions)} {get_shape(actions)} {get_shape(log_prob)} {get_shape(opp_log_prob)}")

                    for i,k in config["OPPONENT"]:
                        actions[k] = opp_actions[k]
                        log_prob[k]=opp_log_prob[k]
                        

                if config.get("HEURISTIC ACTORS",None) is not None:
                    heuristic_actors = config["HEURISTIC_ACTORS"]
                    for a in ["agent_0","agent_1"]:
                        actors = heuristic_actors.get(a,None)
                        if actors is None:
                            continue 
                        rng,_rng = jax.random.split(rng)
                        _rng = jax.random.split(_rng,len(actors))
                        actions = jnp.stack([jax.vmap(a)(env_state.env_state,jax.random.split(r,config["NUM_ENVS"])) for a,r in zip(actors,_rng)]) # 


                new_obs, new_env_state, reward, new_done, info = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )

                rng,rng_opp,rng_heuristic = jax.random.split(rng,3)
                new_opp_ids = jax.random.randint(rng_opp,(config["NUM_ENVS"],),0,jnp.minimum(num_opps,seed_id+1))
                new_heuristic_ids = jax.random.randint(rng_opp,(config["NUM_ENVS"],),0,jnp.minimum(num_heuristic,seed_id+1))
                opp_ids = jnp.where(new_done["__all__"],new_opp_ids,opp_ids)
                heuristic_ids = jnp.where(new_done["__all__"],new_heuristic_ids,heuristic_ids)

                # add shaped reward
                shaped_reward = info.pop("shaped_reward")
                shaped_reward["__all__"] = batchify(shaped_reward).sum(axis=0)
                reward = jax.tree_map(
                    lambda x, y: x + y * rew_shaping_anneal(train_state.timesteps.mean()),
                    reward,
                    shaped_reward,
                )

                reward=jax.tree.map(lambda r:r-config.get("LOGPROB_REW")*batchify(log_prob).mean(),reward) 

                #### MEP and MEPCTOM ####
                mep = config.get("MEP",None)
                if mep is not None:
                    alpha = alpha_anneal(train_state.timesteps.mean())
                    params = jax.lax.all_gather(train_state.params,"seed_axis")
                    batch_stats = jax.lax.all_gather(train_state.batch_stats,"seed_axis") 
                    def get_one_hot_action(p,b,h):
                        h,logits, = jax.vmap(network.apply, in_axes=(0, 0, 0, None,None))(
                            {"params": p, "batch_stats": b},
                            h,
                            batchify(last_obs)[:,None],  # (num_agents, num_envs, num_actions)
                            last_done[None],
                            False,
                        )  # (num_agents, num_envs, num_actions)
                        logits = logits.squeeze(1)
                        logits = logits-(1-batchify(avail_actions))*10000
                        pi = jax.nn.softmax(logits)
                        return h,pi 
                    pop_hs,one_hot_actions = jax.vmap(get_one_hot_action)(params,batch_stats,pop_hs)
                    avg_one_hot_actions = jnp.mean(one_hot_actions,axis=0)
                    ent = distrax.Categorical(probs=avg_one_hot_actions).log_prob(batchify(actions))
                    ent=ent.sum(axis=0)
                    info["mep_logp"]=ent.mean()
                    mep_ent = distrax.Categorical(probs=avg_one_hot_actions).entropy()
                    info["mep_ent_0"]=mep_ent[0].mean()
                    info["mep_ent_1"]=mep_ent[1].mean()
                    reward=jax.tree.map(lambda r:r-alpha*ent,reward) 

                info["shaped_reward"]=shaped_reward["__all__"]
                info["reward_annealing"] = rew_shaping_anneal(train_state.timesteps.mean())
                info["action0_0"]=(batchify(actions)[0]==0).mean()
                info["action0_1"]=(batchify(actions)[1]==0).mean()
                info["action1_0"]=(batchify(actions)[0]==1).mean()
                info["action1_1"]=(batchify(actions)[1]==1).mean()
                info["action2_0"]=(batchify(actions)[0]==2).mean()
                info["action2_1"]=(batchify(actions)[1]==2).mean()
                info["mep_alpha"]=alpha
                info["avg_opp"]=opp_ids.mean()

                # get the next available action
                next_avail_actions = wrapped_env.get_valid_actions(
                    new_env_state.env_state
                )

                transition = Transition(
                    obs=(last_obs),  # (num_agents, num_envs, obs_shape)
                    action=(actions),  # (num_agents, num_envs,)
                    rewards=config.get("REW_SCALE", 1)
                    * reward["__all__"],#[np.newaxis],  # (num_envs,)
                    done=new_done["__all__"],#[np.newaxis],  # (num_envs,)
                    avail_actions=(
                        next_avail_actions
                    ),  # (num_agents, num_envs, num_actions)
                    q_vals=q_vals,  # (num_agents, num_envs, num_actions),
                    state=env_state.env_state,
                    logp=log_prob
                )
                return (new_obs,new_done["__all__"],(hs,critic_hs,opp_hs,pop_hs), (opp_ids,heuristic_ids), new_env_state, rem_advice, teacher_frequencies,teacher_losses, rng), (transition, info)


            # step the env
            # transitions: (num_steps, num_agents, num_envs, ...)
            rng, _rng = jax.random.split(rng)
            (*carry, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, remaining_advice,teacher_frequencies,teacher_losses, _rng),
                None,
                config["NUM_STEPS"],
            )
            remaining_advice,teacher_frequencies,teacher_losses = carry[5:]
            expl_state = tuple(carry[:5])
            

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            last_obs, last_done,(hs,critic_hs,opp_hs,pop_hs),ids, env_state = expl_state
            critic_hs,last_q= critic.apply(
                {
                    "params": critic_train_state.params,
                    "batch_stats": critic_train_state.batch_stats,
                },
                critic_hs,
                last_obs["__all__"][None],  # (num_agents, num_envs, num_actions)
                last_done[None],
                False,
            )  # (num_agents, num_envs, num_actions)
            last_q = last_q.squeeze(0)
            
            def _calculate_gae(values, reward, done, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_q),
                    (done,values,reward),
                    # jax.tree.map(lambda x:jnp.swapaxes(x,0,1),(done,values,reward)),
                    reverse=True,
                )
                return advantages, advantages + values


            if config["NUM_STEPS"] > 1: # q-lambda returns
                advantages, targets = _calculate_gae(transitions.q_vals,transitions.rewards,transitions.done, last_q)
                advantages,targets = jnp.swapaxes(advantages,0,1),jnp.swapaxes(targets,0,1)

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                (train_state,critic_train_state), rng = carry 

                def _learn_phase(carry, minibatch_and_target):

                    # minibatch shape: num_agents, batch_size, ...
                    # target shape: batch_size
                    # with batch_size = num_envs/num_minibatches

                    (train_state,critic_train_state), rng = carry
                    minibatch, adv, targets = jax.tree.map(lambda x:jnp.swapaxes(x,0,1),minibatch_and_target)
                    print(f"single minibatch {jax.tree.map(lambda x:x.shape,minibatch)}")

                    _obs = batchify(minibatch.obs) if isinstance(minibatch.obs,dict) else minibatch.obs
                    _actions = batchify(minibatch.action) if isinstance(minibatch.action,dict) else minibatch.action

                    def _actor_loss_fn(params):
                        batch_size = jax.tree.leaves(minibatch)[0].shape[1]
                        init_hs = MLPLSTM.initialize_carry(network.lstm_dim,env.num_agents,batch_size)
                        print(f"shapes {get_shape(init_hs)} {get_shape(_obs)} {get_shape(minibatch.done)}")
                        (_,logits), updates = jax.vmap(partial(network.apply,train=True,mutable=["batch_stats"]),in_axes=(0,0,0,None))(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            init_hs,
                            _obs,
                            minibatch.done
                        )  # (num_agents*batch_size, num_actions)
                        # print(f"shapes {get_shape(params)} \n\n{get_shape(init_hs)}\n\n {get_shape(train_state.batch_stats)}\n\n{get_shape(_obs)}\n\n{get_shape(minibatch)}")
                        pi = distrax.Categorical(logits=logits)
                        log_prob = pi.log_prob(_actions)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - batchify(minibatch.logp)
                        ratio = jnp.exp(logratio)
                        gae = (adv - adv.mean()) / (adv.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        ent_by_agent = pi.entropy().mean(axis=(1,2))
                        
                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        ent_coef = ent_anneal(train_state.timesteps.mean())
                        actor_loss = (
                            loss_actor
                            - ent_coef * entropy
                        )
                        metrics = {"actor_loss":loss_actor,"entropy_0":ent_by_agent[0],"entropy_1":ent_by_agent[1],"ratio":ratio,"approx_kl":approx_kl,"clip_frac":clip_frac,"ent_coef":ent_coef}
                        return actor_loss, (updates, metrics)
                    

                    (actor_loss, (updates, metrics)), grads = jax.value_and_grad(
                        _actor_loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = jax.vmap(lambda g,t:t.apply_gradients(grads=g))(grads,train_state)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )

                    def _critic_loss_fn(params):
                        # RERUN NETWORK
                        batch_size = jax.tree.leaves(minibatch)[0].shape[1] # seq_len,batch_size
                        init_hs = MLPLSTM.initialize_carry(critic.lstm_dim,batch_size)
                        (_,value), updates = critic.apply(
                            {"params": params, "batch_stats": critic_train_state.batch_stats},
                            init_hs,
                            minibatch.obs["__all__"],
                            minibatch.done,
                            train=True,
                            mutable=["batch_stats"]
                        )  # (num_agents*batch_size, num_actions)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = minibatch.q_vals + (
                            value - minibatch.q_vals
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (updates,{"value_loss":value_loss})

                    (critic_loss, (updates, critic_metrics)), grads = jax.value_and_grad(
                        _critic_loss_fn, has_aux=True
                    )(critic_train_state.params)
                    critic_train_state = critic_train_state.apply_gradients(grads=grads)
                    critic_train_state = critic_train_state.replace(
                        batch_stats=updates["batch_stats"],
                    )

                    metrics = {**metrics,**critic_metrics}

                    return ((train_state,critic_train_state), rng), jax.tree.map(jnp.mean,metrics) 

                def preprocess_transition(x, indices):
                    # x = x.reshape(-1, *x.shape[2:])  # num_steps, num_envs, num_agents
                    x = jnp.swapaxes(x,0,1) # -> num_envs,num_steps,num_agents
                    x = x[indices]
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_minibatches, num_envs/num_minbatches, num_agents, ...
                    return x

                rng, _rng = jax.random.split(rng)
                indices = jax.random.permutation(_rng, config["NUM_ENVS"])  # shuffle the transitions
                minibatches = jax.tree.map(
                    lambda x: preprocess_transition(x, indices),
                    transitions,
                )  # num_minibatches, num_agents, num_envs/num_minbatches ...
                print(f"minibatches {jax.tree.map(lambda x:x.shape,minibatches)}")
                tgt = targets[indices]
                tgt = tgt.reshape(config["NUM_MINIBATCHES"], -1,*tgt.shape[1:])
                gae = advantages[indices]
                gae = gae.reshape(config["NUM_MINIBATCHES"], -1,*gae.shape[1:])

                rng, _rng = jax.random.split(rng)
                ((train_state,critic_train_state), rng), metrics = jax.lax.scan(
                    _learn_phase, ((train_state,critic_train_state), rng), (minibatches, gae,tgt)
                )

                return ((train_state,critic_train_state), rng), jax.tree.map(jnp.mean,metrics)

            if config.get("MODE","train")=="train":
                rng, _rng = jax.random.split(rng)
                ((train_state,critic_train_state), rng), metrics = jax.lax.scan(
                    _learn_epoch, ((train_state,critic_train_state), rng), None, config["NUM_EPOCHS"]
                )
                metrics = jax.tree.map(jnp.mean,metrics)
            else:
                metrics = {}

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics.update({
                "env_step": train_state.timesteps.mean(),
                "update_steps": train_state.n_updates.mean(),
                "grad_steps": train_state.grad_steps.mean(),
                "advice budget":remaining_advice.mean(),
                "seed_id":seed_id,
            })
            metrics.update(jax.tree_map(lambda x: x.mean(), infos))

            # report on wandb if required
            # jax.debug.print("metrics {m}",m=metrics)
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = ((train_state,critic_train_state), save_state, tuple(expl_state), test_state, remaining_advice, transitions, rng)

            return runner_state, metrics


        rng, _rng,rng_opp,rng_heuristic = jax.random.split(rng,4)
        obs, env_state = wrapped_env.batch_reset(_rng)
        hs = opp_hs =  MLPLSTM.initialize_carry(network.lstm_dim,env.num_agents,config["NUM_ENVS"])
        critic_hs =  MLPLSTM.initialize_carry(network.lstm_dim,config["NUM_ENVS"])
        pop_size = jax.lax.all_gather(jnp.array(1),"seed_axis").shape[0]
        pop_hs = MLPLSTM.initialize_carry(network.lstm_dim,pop_size,env.num_agents,config["NUM_ENVS"])
        dones = jnp.zeros((config["NUM_ENVS"],),dtype=jnp.bool_)
        num_opps = config.get("NUM OPPONENTS",1)
        num_heuristic=config.get("NUM HEURISTIC",1)
        opp_ids = jax.random.randint(rng_opp,(config["NUM_ENVS"],),0,jnp.minimum(num_opps,seed_id+1))
        heuristic_ids = jax.random.randint(rng_opp,(config["NUM_ENVS"],),0,jnp.minimum(num_heuristic,seed_id+1))
        expl_state = (obs, dones, (hs,critic_hs,opp_hs,pop_hs), (opp_ids,heuristic_ids), env_state)

        # train
        rng, _rng = jax.random.split(rng)
        remaining_advice = jnp.full((config["NUM_ENVS"]),config.get("ADVICE BUDGET",0))
        runner_state = ((train_state,critic_train_state), (0.,train_state), expl_state, None, remaining_advice, None, _rng)

        rstate_, _ = _update_step(runner_state,None)
        runner_state = runner_state[:-2]+rstate_[-2:]

        runner_state, metrics = jax.lax.scan(
            scan_tqdm(config["NUM_UPDATES"])(_update_step), runner_state, xs=jnp.arange(config["NUM_UPDATES"]), length=config["NUM_UPDATES"]
        )

        sampled_transitions = runner_state[-2]

        def sample_report(x, n_samples=10):
            x_ = x[1:-1]
            n_points = x_.shape[0] 
            x_ = jnp.reshape(x_[:(n_points//n_samples)*n_samples],(n_samples,-1,*x_.shape[1:]))
            x_ =  jnp.mean(x_,axis=1)
            return jnp.concatenate([x[:1],x_,x[-1:]],axis=0) # first, sampled, last

        report_metrics = jax.tree.map(sample_report,metrics)
        return {"runner_state": runner_state, "metrics": report_metrics, "OPPONENT_IDX":OPPONENT_IDX,"sampled_transitions":sampled_transitions}

    return train


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


def env_from_config(config):
    base_name = config["ENV_NAME"]
    kwargs = config['ENV_KWARGS'] 
    env_name = env_name_from_kwargs(base_name,kwargs)
    env = Communication(**kwargs)
    env = LogWrapper(env)
    return env, env_name

def single_run(config, init_params):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    # print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get('ALG_NAME', "pqn_vdn_cnn")
    env, env_name = env_from_config(copy.deepcopy(config))

    def tree_batchify(x: dict):
        return jax.tree.map(lambda *args:jnp.stack(args,axis=0),*[x[agent] for agent in env.agents])

    def tree_unbatchify(x: jnp.ndarray):
        return {agent: jax.tree.map(lambda x_:x_[i],x) for i,agent in enumerate(env.agents)}
    
    if config.get("TEACHER PARAMS",None) is not None:
        config["NUM TEACHERS"]=len(config["TEACHER PARAMS"])
        config["TEACHER PARAMS"] = [tree_batchify(p) for p in config["TEACHER PARAMS"]]
        config["TEACHER PARAMS"] = jax.tree.map(lambda *args:jnp.stack(args,axis=0),*config["TEACHER PARAMS"])
    if config.get("OPPONENT PARAMS",None) is not None:
        config["NUM OPPONENTS"]=len(config["OPPONENT PARAMS"])
        config["OPPONENT PARAMS"] = [tree_batchify(p) for p in config["OPPONENT PARAMS"]]
        config["OPPONENT PARAMS"] = jax.tree.map(lambda *args:jnp.stack(args,axis=0),*config["OPPONENT PARAMS"])
    if config.get("INIT PARAMS",None) is not None:
        config["INIT PARAMS"] = tree_batchify(config["INIT PARAMS"])
        pass 
    if config.get("HEURISTIC ACTORS",None) is not None:
        config["NUM HEURISTIC"] = len(config["HEURISTIC ACTORS"])


    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f"{alg_name}_{env_name}",
        config={k:v for k,v in config.items() if k not in {"TEACHER PARAMS", "OPPONENT PARAMS", "HEURISTIC ACTOR"}},
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    if init_params is not None:
        init_params = [tree_batchify(p) for p in init_params]
        init_params = jax.tree.map(lambda *args:jnp.stack(args,axis=0),*init_params)


    def scan_loop(f,arg):
        s = min(config["NUM_SEEDS"],25)
        arg = jax.tree.map(lambda x:jnp.reshape(x,(-1,s,*x.shape[1:])),arg)
        _,res = jax.lax.scan(lambda a,b:(a,jax.vmap(f,axis_name="seed_axis")(b)),0,xs=arg)
        res = jax.tree.map(lambda x:jnp.reshape(x,(-1,*x.shape[2:])),res)
        return res 
    def get_res(rng):
        return scan_loop(make_train(config,env),rng)
    outs = jax.block_until_ready(jax.jit(get_res)((rngs,jnp.arange(config["NUM_SEEDS"])%config.get("NUM OPPONENTS",1),None)))
    

    jnp.set_printoptions(linewidth=300)
    import pprint 
    pprint.pprint(f"opponents: {outs['OPPONENT_IDX']} \nfinal metrics:")
    
    ignore = ["actor_loss","advice budget","clip_frac","env_step","grad_steps","ratio","returned_episode","returned_episode_lengths",
              "reward_annealing","shaped_reward","update_steps","value_loss","approx_kl"]
    final_metrics = {k:v for k,v in outs["metrics"].items() if k not in set(ignore)}
    pprint.pprint(final_metrics)

    with open(f"catdog_cache/{config.get('CACHE_NAME','cache')}.json","w") as f:
        json.dump({k:v.tolist() for k,v in final_metrics.items()},f)
    # save params
    if config.get("SAVE_PATH", None) is not None:
        model_state = outs["runner_state"][0][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        config = {k:v for k,v in config.items() if k not in {"TEACHER PARAMS", "OPPONENT PARAMS", "HEURISTIC ACTOR", "INIT PARAMS"}}
        os.makedirs(os.path.join(
                save_dir, f'{alg_name}_{env_name}'
            ), exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}/seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = {"params":jax.tree_map(lambda x: x[i], model_state.params), "batch_stats":jax.tree_map(lambda x: x[i], model_state.batch_stats)}
            params = tree_unbatchify(params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}/seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


@hydra.main(version_base=None, config_path="./config", config_name="catdog_config")
def main(config):
    config = OmegaConf.to_container(config)
    
    config["RESPONSE"]=[(1,"agent_1")]
    config["OPPONENT"]=[(0,"agent_0")]

    config["ADVICE BUDGET"]=500000
    config["ADVISE PROB"]=0.75 #0.5
    config["ADVICE DECAY"]=0.01**(1/(3e6/64/16))

    config["ADVISE CONCEPT THRESHOLD"]=0.5
    config["OPPONENT SELECTION"]="random once"

    config["ADVICE MODE"]=["actions"]
    config["CTOM"] = True
    config["NETWORK_USES_CTOM"] = False 

    config["CTOM_VERSION"]=5
    config["ACTION_ANTICIPATION"]=4
    config["LOOKAHEAD"]=float("inf")
    config["CTOM_ACTIVATION"]="SIGMOID"
    config["PREDICTOR_LR"]=0.0003
    config["PREDICTOR LR"]=0.0003
    config["PREDICTOR_BATCH_SIZE"]=2

    # config["MEP"]="ACTIONS"
    # config["MEP_ALPHA"]=10
    # config["MEP_ALPHA_HORIZON"]=1
    config["LOGPROB_REW"]=0
    config["GAE_LAMBDA"]= 0.95
    config["CLIP_EPS"]= 0.2
    config["SCALE_CLIP_EPS"]= False
    config["ENT_COEF"]=   1 # 0.1
    config["ENT_FINISH"]= 0.01
    config["ENT_HORIZON"]= 0.75 # 0
    config["VF_COEF"]= 0.5
    config["BR"]=False
    config["MODE"]="train"
    config["CACHE_NAME"]="br"


    pop_env_name =  env_name_from_kwargs(config["alg"]["ENV_NAME"],{**config["alg"]["ENV_KWARGS"],**{"max_games":1}})

    print("Config:\n", OmegaConf.to_yaml(config))

    def get_name(f,i=0,seed=154):
        return f"/home/acni/pworld/save/catdog/{pop_env_name}/{f}_{pop_env_name}/seed{seed}_vmap{i}.safetensors"
    filtered_vmap = list(range(25))
    opponent_params = [get_name(f"br_adaptation-reveal_class-train_teacher",i=i) for i in filtered_vmap]

    config["OPPONENT PARAMS"]=[load_params(t) for t in opponent_params]

    single_run(config,None)


if __name__ == "__main__":
    main()