import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

class GroupLinearLayer(nn.Module):
    din: int
    dout: int
    num_blocks: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        # x is expected to be shape: [batch_size, num_blocks, din]
        
        w = self.param(
            "w",
            nn.initializers.lecun_normal(),
            (self.num_blocks, self.din, self.dout),
        )
        
        # bnd = [batch, blocks, din], ndo = [blocks, din, dout] -> bno = [batch, blocks, dout]
        y = jnp.einsum("bnd,ndo->bno", x, w)
        
        if self.use_bias:
            b = self.param(
                "b",
                nn.initializers.zeros,
                (self.num_blocks, self.dout),
            )
            # JAX automatically broadcasts 'b' from (blocks, dout) to (batch, blocks, dout)
            y = y + b
            
        return y

class GroupGRUCell(nn.Module):
    input_size: int
    hidden_size: int
    num_grus: int

    @nn.compact
    def __call__(self, x, hidden):
        x2h = GroupLinearLayer(
            self.input_size, 3 * self.hidden_size, self.num_grus, name="x2h"
        )
        h2h = GroupLinearLayer(
            self.hidden_size, 3 * self.hidden_size, self.num_grus, name="h2h"
        )

        gate_x = x2h(x)
        gate_h = h2h(hidden)

        i_r, i_i, i_n = jnp.split(gate_x, 3, axis=-1)
        h_r, h_i, h_n = jnp.split(gate_h, 3, axis=-1)

        resetgate = nn.sigmoid(i_r + h_r)
        inputgate = nn.sigmoid(i_i + h_i)
        newgate = jnp.tanh(i_n + resetgate * h_n)

        hy = newgate + inputgate * (hidden - newgate)
        return hy



class DenseModularCell(nn.Module):
    input_size: int
    hidden_size: int
    num_units: int
    comm_key_size: int = 32
    comm_query_size: int = 32
    num_comm_heads: int = 16

    def setup(self):
        comm_value_size = self.hidden_size

        self.rnn = GroupGRUCell(
            self.input_size,
            self.hidden_size,
            self.num_units,
            name="group_gru",
        )

        self.query_ = GroupLinearLayer(
            self.hidden_size,
            self.num_comm_heads * self.comm_query_size,
            self.num_units,
            name="comm_query",
        )
        self.key_ = GroupLinearLayer(
            self.hidden_size,
            self.num_comm_heads * self.comm_key_size,
            self.num_units,
            name="comm_key",
        )
        self.value_ = GroupLinearLayer(
            self.hidden_size,
            self.num_comm_heads * comm_value_size,
            self.num_units,
            name="comm_value",
        )
        self.comm_attention_output = GroupLinearLayer(
            self.num_comm_heads * comm_value_size,
            comm_value_size,
            self.num_units,
            name="comm_out",
        )

    def transpose_for_scores(self, x, num_heads, head_dim):
        b, n, _ = x.shape
        x = x.reshape(b, n, num_heads, head_dim)
        return jnp.transpose(x, (0, 2, 1, 3))

    def communication_attention(self, h):
        query_layer = self.query_(h)
        key_layer = self.key_(h)
        value_layer = self.value_(h)

        query_layer = self.transpose_for_scores(
            query_layer, self.num_comm_heads, self.comm_query_size
        )
        key_layer = self.transpose_for_scores(
            key_layer, self.num_comm_heads, self.comm_key_size
        )
        value_layer = self.transpose_for_scores(
            value_layer, self.num_comm_heads, self.hidden_size
        )

        attention_scores = jnp.matmul(
            query_layer, jnp.swapaxes(key_layer, -1, -2)
        ) / math.sqrt(self.comm_key_size)

        attention_probs = nn.softmax(attention_scores, axis=-1)

        context = jnp.matmul(attention_probs, value_layer)
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = context.reshape(context.shape[0], context.shape[1], -1)

        context = self.comm_attention_output(context)
        return context + h

    @nn.compact
    def __call__(self, hs, x):
        b = x.shape[0]

        # same input to every module
        inputs = jnp.broadcast_to(
            x[:, None, :],
            (b, self.num_units, x.shape[-1]),
        )

        hs = self.rnn(inputs, hs)
        # hs = self.communication_attention(hs)

        y = hs.reshape(hs.shape[0], -1)
        return hs, y

    @staticmethod
    def initialize_carry(batch_size, num_units, hidden_size):
        return jnp.zeros((batch_size, num_units, hidden_size), dtype=jnp.float32)


# OLD CODE
# class GroupLinearLayer(nn.Module):
#     din: int
#     dout: int
#     num_blocks: int

#     @nn.compact
#     def __call__(self, x):
#         w = self.param(
#             "w",
#             nn.initializers.normal(stddev=0.01),
#             (self.num_blocks, self.din, self.dout),
#         )
#         return jnp.einsum("bnd,ndo->bno", x, w)

