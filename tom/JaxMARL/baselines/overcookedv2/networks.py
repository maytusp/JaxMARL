import functools
from typing import Any, Callable, Dict, NamedTuple, Sequence

import distrax
import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    rnn_type: str = None

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        h, w, c = obs.shape[-3:]
        flat_obs = obs.reshape(-1, h, w, c)
        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = embed_model(flat_obs)
        embedding = embedding.reshape(*obs.shape[:-3], -1)
        embedding = nn.LayerNorm()(embedding)

        hidden, embedding = ScannedRNN(config=self.config, rnn_type=self.rnn_type)(hidden, (embedding, dones))

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)



class ScannedRNN(nn.Module):
    config: Dict = None
    rnn_type: str = None

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        current_rnn_type = self.rnn_type if self.rnn_type is not None else self.config.get("RNN_TYPE", "gru")
        current_batch_size = ins.shape[0]

        if current_rnn_type != "gru":
            raise ValueError("This perspective-aux FCP script currently supports GRU only.")

        hidden_dim = self.config["GRU_HIDDEN_DIM"]
        new_carry = self.initialize_carry(self.config, rnn_type="gru", batch_size=current_batch_size)
        rnn_state = jnp.where(resets[:, None], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=hidden_dim)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(config, rnn_type=None, batch_size=None):
        if batch_size is None:
            batch_size = config["NUM_ENVS"]
        hidden_dim = config["GRU_HIDDEN_DIM"]
        cell = nn.GRUCell(features=hidden_dim)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_dim))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = False):
        if x.ndim == 3:
            x = x[jnp.newaxis, ...]
            had_no_batch = True
        else:
            had_no_batch = False

        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        if had_no_batch:
            x = x.squeeze(0)
        return x
