import jax 
import os 
import re 
import json
import jax.numpy as jnp 
import numpy as np
import distrax 
from scipy.stats import bootstrap 
from jax_tqdm import scan_tqdm
import copy 
import jaxmarl
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from baselines.tomzsc.utils.wrappers import LogWrapper, load_params


CPU = jax.devices("cpu")[0]
GPU = jax.devices("gpu")[0]

def get_mean_ci(x,metric=np.mean):
    x = np.float32(x)
    if len(x)==1:
        return [np.mean(x).item()]*3
    mean = metric(x).item()
    ci = bootstrap((x,),metric).confidence_interval 
    low = np.min(x).item() if np.isnan(ci.low) else ci.low 
    high = np.max(x).item() if np.isnan(ci.high) else ci.high
    return [mean,low,high]

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
    leaves = jax.tree.leaves(x)
    if len(leaves)==0:
        return 0
    return jax.tree.leaves(x)[0].shape[0]

def random_argmin(key,x,threshold=0.01):
    under_threshold = x<=(jnp.min(x)+threshold)
    logits = jnp.where(under_threshold,0,-10000) 
    pi = distrax.Categorical(logits=logits) 
    return pi.sample(seed=key)

def scan_loop(f,s=1,add_vmap=True,track=False):
    'Similar syntax to jax.vmap, except instead of vectorizing over the batch dimension, scans over the batch dimension in chunks instead.'
    f = jax.vmap(f) if add_vmap else f 
    def wrapper(*arg):
        arg = jax.tree.map(lambda x:jnp.reshape(x,(-1,s,*x.shape[1:])),arg)
        if track:
            n_steps = get_batch_size(arg)
            _,res = jax.lax.scan(scan_tqdm(n_steps)(lambda a,b:(a,f(*b[1]))),0,xs=(jnp.arange(n_steps),arg))
        else:
            _,res = jax.lax.scan(lambda a,b:(a,f(*b)),0,xs=(arg))
        res = jax.tree.map(lambda x:jnp.reshape(x,(-1,*x.shape[2:])),res)
        return res 
    return wrapper

def convert_clusters_to_labels(clusters):
    'Convert a list of clusters, where each cluster is a list of integers, to an array of cluster labels'
    total = sum([len(x) for x in clusters])
    ret = np.zeros(total,dtype=int)
    for i,x in enumerate(clusters):
        ret[np.array(x)]=i 
    return ret


def env_from_config(config):
    env_name = config["ENV_NAME"]
    layout_name = config["ENV_KWARGS"]["layout"]
    env_name_full = f"{env_name}_{layout_name}"

    env = jaxmarl.make(env_name, **config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)
    return env, env_name_full



def get_agent_parameter_paths(save_dir: str):
    if save_dir is None:
        return None
    files = sorted([f for f in os.listdir(save_dir) if re.match(r"seed\d+_vmap\d+\.safetensors",f)])
    paths = [os.path.join(save_dir, f) for f in files]
    return paths

def load_from_paths(paths: list):
    if paths is None:
        return None
    params = [tree_batchify(load_params(p)) for p in paths]
    params = jax.tree.map(lambda *args:jnp.stack(args),*params)
    return params

def get_param_paths(config: dict, group_teammates=True):
    '''
    Get the paths to each set of saved parameters for the teammate agents and maybe ego agents, if specified, from their save directories
    Optionally provide cluster assignments to group teammates into clusters. Also returns cluster labels
    '''
    # Get teammate paths, optionally group into clusters
    teammate_paths = get_agent_parameter_paths(config["TEAMMATE_DIR"])

    cluster_labels = config.get("CLUSTER_LABELS")
    if cluster_labels is not None:
        # Optionally provide json file instead of explicitly specifying clusters
        if isinstance(cluster_labels, str):
            with open(cluster_labels, "r") as f:
                cluster_labels = json.load(f)
        unique_labels = np.unique(cluster_labels)
        if group_teammates:
            teammate_paths = [[p for p,l_ in zip(teammate_paths,cluster_labels) if l_==l] for l in unique_labels]

    # Optionally get ego paths
    ego_paths = get_agent_parameter_paths(config.get("EGO_DIR"))

    # Optionally get training partner paths
    training_paths = get_agent_parameter_paths(config.get("TRAINING_DIR"))
    
    return teammate_paths, ego_paths, training_paths, cluster_labels


def get_tom_paths(config: dict):
    tom_dir = config.get("TOM_DIR")
    if tom_dir is None:
        return [None, None]
    files = ["cluster_tom.safetensors", "global_tom.safetensors"]
    paths = [os.path.join(tom_dir, f) for f in files]
    return paths


def construct_coarse_concepts(concepts: jnp.ndarray, super_coarse = False) -> jnp.ndarray:
    "Quick and dirty manual coalescing of concepts from a fine concept set to a coarse or super coarse concept set"
    res = jnp.maximum(concepts[...,1:16],concepts[...,16:31])
    res = jnp.maximum(res,concepts[...,31:46])
    res = jnp.maximum(res,concepts[...,46:61])
    res = jnp.concat([concepts[...,:1],res],axis=-1)

    if super_coarse:
        _pickup = jnp.maximum(concepts[...,1:5],concepts[...,5:9])
        _place = jnp.maximum(concepts[...,9:11], concepts[...,11:13])
        _serve = jnp.maximum(concepts[...,14:15], concepts[...,15:16])
        res = jnp.concat([res[...,:1], _pickup, _place, concepts[...,13:14], _serve],axis=-1)
    return res 

def get_concept_shape(concept_set: str):
    if concept_set == "actions":
        concept_shape = (2,6) 
        concept_dim = 2*6 
    elif concept_set == "coarse":
        concept_shape = (2,16)
        concept_dim = 2*16
    elif concept_set == "super_coarse":
        concept_shape = (2,9)
        concept_dim = 2*9
    elif concept_set == "fine":
        concept_shape = (2, 61)
        concept_dim = 122
    else:
        raise ValueError(f"No such concept set specification: {concept_set}")
    return concept_shape, concept_dim
