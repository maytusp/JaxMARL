# ToMZSC
Code for the paper "Theory of Mind Guided Strategy Adaptation for Zero-Shot Coordination"


## Get Started

### Install

Create a python environment with the required dependencies, mainly env-related (gymnax, jaxmarl), training-related (numpy, jax + related packages, e.g. flax, optax), logging/config related (hydra, omegaconf, wandb, safetensors), and auxiliary scientific functions (scipy, sklearn) using

```bash
pip install -r requirements.txt
```

### Run

- `overcooked_train_rl.py` includes all of the logic for RL training, including training the training pool of teammate agents, the held-out evaluation pool, as well as the best-response policies. Example usage: 

```bash
python overcooked_train_rl.py \
    +alg="overcooked_teammate" \
    alg.ENV_KWARGS.layout="large_room" \
    NUM_SEEDS=10
```

- `overcooked_cross_play.py` includes logic for rolling out pretrained policies, e.g. evaluating self-play and cross-play score matrices or collecting trajectories for ToM training. Example usage: 

```bash
python overcooked_cross_play.py \
    +alg="overcooked_cross_play" \
    alg.ENV_KWARGS.layout="large_room" \
    alg.CROSS_PLAY_MODE="cross_play" \
    alg.TEAMMATE_DIR="path/to/training_pool_safetensors/" 
```


- `clustering/get_clusters.py` includes logic for computing cluster labels from self play and cross play scores. Example usage:

```bash
python clustering/get_clusters.py \
    --infile "path/to/cross_play_matrix.json" \
    --outdir "path/to/output/directory"
```

- `overcooked_train_tom.py` includes logic for training ToM networks. Example usage:

```bash
python overcooked_train_tom.py \
    +alg="overcooked_train_tom" \
    alg.ENV_KWARGS.layout="large_room" \
    alg.TEAMMATE_DIR="path/to/training_pool_safetensors/" \
    alg.EGO_DIR="path/to/cluster_br_safetensors" \
    alg.CLUSTER_LABELS="path/to/clusters.json"
```



