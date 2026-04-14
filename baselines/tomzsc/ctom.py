import chex
import optax
import jax
import jax.numpy as jnp
from jax import vmap 
from typing import Any,Union

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

@chex.dataclass()
class Timestep:
    state: dict
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict
    concepts: Union[dict,None] = None 
    tom_outputs: Union[dict,None] = None 
    ground_truth_tom: Union[dict,None] = None

    def flatten(self):
        reward_ndim = jax.tree.leaves(self.rewards)[0].ndim 
        get_shape = lambda x:(-1,*x.shape[reward_ndim:])
        return jax.tree.map(lambda x:jnp.reshape(x,get_shape(x)),self)


def bce(x,y,reduction="mean") -> jnp.ndarray:
    "Binary cross entropy"
    x = jnp.clip(x,min=1e-8,max=1-1e-8)
    res = y*jnp.log(x)+(1-y)*jnp.log(1-x)
    if reduction=="mean":
        return -jnp.mean(res)
    elif reduction=="none":
        return -res 
    else:
        raise NotImplementedError(f"no such reduction {reduction}")


def bkl(x,y,reduction="mean") -> jnp.ndarray:
    "Binary KL Divergence"
    return bce(x,y,reduction=reduction)-bce(y,x,reduction=reduction)


def make_tom_target(tom_outputs, ground_truth, i, self_tom_mode="i"):
    "Make the recursive tom target for the ith individual"
    assert self_tom_mode in ["i", 0], "Self tom mode can only be \"i\" or 0"

    output_dims = ground_truth.ndim+1  if tom_outputs is None else tom_outputs.ndim
    gt_dim = ground_truth.ndim 

    if output_dims<=gt_dim:
        output = jnp.int32(0)
    if output_dims<=gt_dim+1:
        output = ground_truth
    else:
        ### Agent i's self-tom is in the ith position
        if self_tom_mode == "i":
            n = tom_outputs.shape[0]
            output = jnp.zeros(tom_outputs.shape[1:])
            indices = jnp.mod(i+jnp.arange(1,n),n)
            output = output.at[indices].set(tom_outputs[(indices,indices)])
            output = output.at[i].set(make_tom_target(tom_outputs[(jnp.arange(n),jnp.arange(n))],ground_truth,i))
        ### Agent i's self-tom is in the 0th position
        else:
            n = tom_outputs.shape[0]
            output = jnp.zeros(tom_outputs.shape[1:])
            indices = jnp.mod(i+jnp.arange(1,n),n)
            output = output.at[jnp.arange(1,n)].set(tom_outputs[(indices,jnp.zeros_like(indices))])
            output = output.at[0].set(make_tom_target(tom_outputs[(jnp.arange(n),jnp.zeros((n,),dtype=jnp.int32))],ground_truth,i))

    return output 


def common(env):
    "Defines and returns some common utility functions dependent on the environment"
    def batchify(x: dict):
        return x if x is None else jnp.stack([x[agent] for agent in env.agents], axis=0)
    def unbatchify(x: jnp.ndarray):
        return x if x is None else {agent: x[i] for i, agent in enumerate(env.agents)}
    def pad_concepts(concepts, lookahead, axis=0):
        ndim = concepts.ndim 
        pad = ((0,0),)*axis + ((0,lookahead),) + ((0,0),)*(ndim-axis-1)
        return jnp.pad(concepts, pad, mode="edge")
    def all_agent_concepts(f):
        def wrapper(s):
            # print(f"state shape {jax.tree.map(lambda x:x.shape,s)}")
            return vmap(f,in_axes=(0,None))(jnp.arange(env.num_agents),s)
        return wrapper
    def process_concept_fn(f):
        agent_vmap = all_agent_concepts(f)
        return vmap(vmap(agent_vmap)) # Vmap over steps and envs
    make_all_targets = lambda output,gt:vmap(make_tom_target,in_axes=(None,0,0))(output,gt,jnp.arange(env.num_agents))
    return batchify, unbatchify,pad_concepts,process_concept_fn,make_all_targets


def overcooked_ctom(env, ctom_activation="SIGMOID", concept_dim=61,lookahead=float("inf"), **kwargs):
    batchify,unbatchify,pad_concepts,process_concept_fn,make_tom_target = common(env)
    softmax_loss = lambda x,y,reduction="none":optax.softmax_cross_entropy(jnp.log(x).reshape([*x.shape[:-1],2,x.shape[-1]//2]),y.reshape([*y.shape[:-1],2,y.shape[-1]//2]),axis=-1).mean(axis=-1 if reduction=="none" else None)
    loss_fn = bce if ctom_activation=="SIGMOID" else softmax_loss
    def one_hot(x):
        n_concepts = concept_dim
        return jax.lax.cond(x>=0,lambda:jax.nn.one_hot(x,n_concepts),lambda:jnp.zeros(n_concepts))
    def _get_concept(aidx, trajectory): # (n_steps) -> (n_steps)
        def scan_fn(carry,timestep):
            my_prev_interact,other_prev_interact,my_timesteps,other_timesteps=carry # what 

            my_interact = timestep.state.agent_interact[aidx]
            other_interact = timestep.state.agent_interact[1-aidx]

            my_timesteps = jnp.where(my_interact!=0,lookahead,my_timesteps-1)
            my_timesteps = jnp.where(timestep.done,lookahead,my_timesteps)
            other_timesteps = jnp.where(other_interact!=0,lookahead,other_timesteps-1)
            other_timesteps = jnp.where(timestep.done,lookahead,other_timesteps)
            
            my_interact = jnp.where(my_interact==0,my_prev_interact,my_interact)
            my_interact = jnp.where(timestep.done | (my_timesteps<0) ,my_interact*0,my_interact)

            other_interact = jnp.where(other_interact==0,other_prev_interact,other_interact)
            other_interact = jnp.where(timestep.done | (other_timesteps<0), other_interact*0,other_interact)

            ret = jnp.stack([one_hot(my_interact),one_hot(other_interact)],axis=0) 
            return (my_interact,other_interact,my_timesteps,other_timesteps),(ret,my_interact)
        _,(concepts,res) = jax.lax.scan(scan_fn,(-1,-1,lookahead,lookahead),trajectory,reverse=True) # n_steps,n_agents,40

        return concepts
    def get_agent_concepts(traj_batch): # traj_batch = (num_steps, num_envs)
        concepts = jax.vmap(jax.vmap(_get_concept,in_axes=(None,1),out_axes=1),in_axes=(0,None),out_axes=2)(jnp.arange(env.num_agents),traj_batch) # (n_steps,n_envs,n_agents)
        return concepts 
    def _process_trajectory(trajectory): #(num_steps, num_envs, (num_agents?))
        concepts = get_agent_concepts(trajectory) # (num_steps, num_envs, num_agents, concept_dim)
        concepts_available = jnp.where(jnp.sum(concepts,axis=range(2,concepts.ndim))<0.9,0.,1.)
        trajectory = trajectory.replace(concepts=concepts,concepts_available=concepts_available)
        ground_truth = vmap(vmap(lambda t:make_tom_target(batchify(t.tom_outputs),t.concepts)))(trajectory)
        trajectory = trajectory.replace(ground_truth_tom=ground_truth)
        return trajectory
    def get_concept_losses(tom_output,state, reduction="mean"):
        concepts = vmap(lambda i:env.get_concept(state,i))(jnp.arange(env.num_agents))
        ground_truth = make_tom_target(tom_output,concepts)
        res = vmap(lambda p,t:bce(p,t))(tom_output,ground_truth)
        if reduction=="mean":
            return vmap(jnp.mean)(res) 
        elif reduction=="none":
            return res 
        else:
            raise NotImplementedError(f"no such reduction {reduction}")
    return _process_trajectory, loss_fn, get_concept_losses


def get_ctom(env,config):
    ctom_activation = config.get("CTOM_ACTIVATION","SIGMOID")
    ctom_lookahead = config.get("LOOKAHEAD",float("inf"))
    _process_trajectory,get_concept_loss, teacher_concept_loss = overcooked_ctom(env, ctom_activation=ctom_activation, concept_dim=61,lookahead=ctom_lookahead)
    
    return _process_trajectory, get_concept_loss, teacher_concept_loss



        



