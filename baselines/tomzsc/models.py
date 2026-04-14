import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from functools import partial
from flax.linen.initializers import constant, orthogonal


class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )
    

class ScannedLSTM(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]

        rnn_state = jax.tree.map(lambda x,y:jnp.where(resets[:, np.newaxis],x,y),self.initialize_carry(hidden_size, *ins.shape[:-1]),rnn_state)
        new_rnn_state, y = nn.OptimizedLSTMCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.LSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class BaseNetwork(nn.Module):
    action_dim: int
    init_scale: float = 1.0
    body_dims: tuple = (64,64,64)
    head_dims: tuple = (64,64)
    rnn_dim: int = 64

    @nn.compact
    def __call__(self, hidden, obs, dones):
        x = obs 
        for dim in self.body_dims:
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)

        rnn_in = (x, dones)
        hidden, x = ScannedRNN()(hidden, rnn_in)

        for dim in self.body_dims:
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)

        return hidden, q_vals
    
class RecurrentNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    output_dim: int
    hidden_dims: tuple = (64,64,64)
    rnn_dim: int = 64
    init_scale: float = 1.0
    use_lstm: bool=True 

    @nn.compact
    def __call__(self, hidden, obs, dones):
        x = obs 
        for dim in self.hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)

        rnn_in = (x, dones)
        if self.use_lstm:
            hidden, x = ScannedRNN()(hidden, rnn_in)
        else:
            x = nn.Dense(self.rnn_dim)(x)

        output = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)

        return hidden, output
    
    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        res = nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )
        return res
    

class OvercookedPredictConcepts(nn.Module):
    norm_type: str = "layer_norm"
    num_layers: int = 2
    # norm_input: bool = False 
    hidden_size: int = 64
    rnn_dim: int = 64
    concept_dim: int = None
    concept_shape: tuple = None
    use_lstm: bool=True

    @nn.compact
    def __call__(self, hidden, x, dones, train=False):
        # print(f"hidden {jax.tree.map(lambda x:x.shape,hidden)} x {jax.tree.map(lambda x:x.shape,x)} done {jax.tree.map(lambda x:x.shape,dones)}")
        # exit(0)

        activation = nn.relu

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        # if self.norm_input:
        #     x = nn.BatchNorm(use_running_average=not train)(x)
        # else:
        #     x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = x.reshape((*x.shape[:-3], -1))  # Flatten

        hidden,x = RecurrentNetwork(self.concept_dim,use_lstm=self.use_lstm)(hidden,x,dones)

        x = jax.nn.sigmoid(x)
        if self.concept_shape is not None:
            x = jnp.reshape(x,(*x.shape[:-1],*self.concept_shape))
        
        return hidden,x
    
    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        res = nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )
        return res
        

class OvercookedPQNCNN(nn.Module):
    norm_type: str = "layer_norm"
    output_dim: int = 32
    hidden_dims: int = 32

    @nn.compact
    def __call__(self, x, train=False):

        activation = nn.relu

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(
            features=self.hidden_dims,
            kernel_size=(5, 5),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Conv(
            features=self.hidden_dims,
            kernel_size=(3, 3),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Conv(
            features=self.output_dim,
            kernel_size=(3, 3),
        )(x)

        return x


class OvercookedPQNCNNResidual(nn.Module):
    norm_type: str = "none"

    @nn.compact
    def __call__(self, x):
        return OvercookedPQNCNN(self.norm_type,output_dim=32)(x)

class OvercookedPQNCNNConcept(nn.Module):
    norm_type: str = "none"
    concept_dim: int = 1

    @nn.compact
    def __call__(self, x):
        x = OvercookedPQNCNN(self.norm_type,output_dim=self.concept_dim)(x)
        return jax.nn.softmax(x,axis=(-2,-3)) #jax.nn.sigmoid(x)

class OvercookedAgentNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64
    num_layers: int = 2
    norm_input: bool = False
    norm_type: str = "layer_norm"
    activation: str = "relu"
    concept_dim: int = 122 
    agent_action_dims: tuple = tuple()
    ctom_version: int = 2
    ctom_activation: str = "SIGMOID"
    concept_shape: tuple = (2,61)
    use_ctom: bool = False 
    featurized: bool = False
    action_anticipation_level: int = 0

    @nn.compact
    def __call__(self, obs, use_concepts=False, concepts=None, concept_coef=1, train=False):

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.norm_input:
            obs = nn.BatchNorm(use_running_average=not train)(obs)
        else:
            x_dummy = nn.BatchNorm(use_running_average=not train)(obs)

        concept_output = OvercookedPQNCNNResidual(self.norm_type,name="concept_conv")(obs)
        concept_output = activation(normalize(concept_output))
        concept_output = concept_output.reshape((*concept_output.shape[:-3], -1))
        concept_output = nn.Dense(features=64,name="concept_fc1")(concept_output)
        concept_output = activation(normalize(concept_output))
        concept_output = nn.Dense(features=64,name="concept_fc2")(concept_output)
        concept_output = activation(normalize(concept_output))
        concept_output = nn.Dense(features=self.concept_dim,name="concept_output")(concept_output)

        def process_softmax(x):
            return jax.nn.softmax(x.reshape(*x.shape[:-1],2,x.shape[-1]//2),axis=-1).reshape(x.shape)
        
        concept_output = jax.nn.sigmoid(concept_output) if self.ctom_activation=="SIGMOID" else process_softmax(concept_output)

        residuals = OvercookedPQNCNNResidual(self.norm_type)(obs)
        residuals = activation(normalize(residuals))
        residuals = residuals.reshape((*residuals.shape[:-3], -1)) 
        residuals = nn.Dense(features=64)(residuals)
        residuals = activation(normalize(residuals))
        residuals = nn.Dense(features=64)(residuals)
        

        if use_concepts:
            concepts = concepts*concept_coef + concept_output * (1-concept_coef)
        else: 
            concepts = concept_output

        if self.use_ctom:
            bottleneck = jnp.concatenate([concepts,residuals],axis=-1) 
        else:
            bottleneck = residuals


        x = nn.Dense(features=64)(bottleneck)
        x = activation(normalize(x))

        x = nn.Dense(features=64)(x)
        x = activation(normalize(x))

        q_values = nn.Dense(features=self.action_dim)(x)


        if self.concept_shape is not None:
            concept_output = jnp.reshape(concept_output,(*concept_output.shape[:-1],*self.concept_shape))

        return (q_values, concept_output)
    
class JaxMARLCNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x, train=False):

        activation = nn.relu

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = normalize(x)
        x = activation(x)

        return x

class JaxMARLLSTM(nn.Module):
    action_dim: int
    hidden_size: int = 64
    rnn_dim: int = 64
    num_layers: int = 2
    norm_input: bool = False
    norm_type: str = "layer_norm"
    use_lstm: bool = True

    @nn.compact
    def __call__(self, hidden, x: jnp.ndarray, dones, train=False):

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        x = JaxMARLCNN(norm_type=self.norm_type)(x, train=train)

        x = x.reshape((*x.shape[:2], -1))  # Flatten to (bsz,seq_len,dim)
        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        if self.use_lstm:
            hidden, x = ScannedRNN()(hidden, (x, dones))
        else:
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        q_vals = nn.Dense(self.action_dim)(x)
        
        return hidden, q_vals
    
    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )
    

class JaxMARLLSTMGCRL(nn.Module):
    action_dim: int
    hidden_size: int = 64
    rnn_dim: int = 64
    num_layers: int = 2
    norm_input: bool = False
    norm_type: str = "layer_norm"
    use_lstm: bool = True
    num_concepts: int = 3

    @nn.compact
    def __call__(self, hidden, x: jnp.ndarray, dones, train=False):
        concept_hidden, actor_hidden = hidden 

        concept_hidden, concept_output = JaxMARLLSTM(self.num_concepts,self.hidden_size,self.rnn_dim,self.num_layers,self.norm_input,self.norm_type,self.use_lstm)(concept_hidden,x,dones,train=train)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        x = JaxMARLCNN(norm_type=self.norm_type)(x, train=train)

        x = x.reshape((*x.shape[:2], -1))  # Flatten to (bsz,seq_len,dim)
        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        if self.use_lstm:
            actor_hidden, x = ScannedRNN()(actor_hidden, (x, dones))
        else:
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = jnp.concatenate([x,jax.lax.stop_gradient(concept_output)],axis=-1)

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        q_vals = nn.Dense(self.action_dim)(x)
        
        return (concept_hidden, actor_hidden), (q_vals, concept_output)
    
    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        hidden = nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )
        return (hidden,hidden)
    
class MLP(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    output_dim: int
    hidden_dims: tuple = (64,64,64)
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, obs):
        x = obs 
        for dim in self.hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)

        output = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)

        return output

class MLPLSTM(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    output_dim: int
    input_dims: tuple = (64,64,64)
    lstm_dim: int = 64
    output_dims: tuple = (64,64,64)
    init_scale: float = 1.0
    use_lstm: bool = True 
    critic: bool=False 
    no_batchnorm: bool=False
    use_sigmoid: bool = False
    concept_shape: tuple=None 

    @nn.compact
    def __call__(self, hidden, x, dones,train=False): 
        if not self.no_batchnorm:
            dummy = nn.BatchNorm(use_running_average=not train)(x)
        for dim in self.input_dims:
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)

        if self.use_lstm:
            hidden, x = ScannedLSTM()(hidden, (x, dones))
        else:
            x = nn.Dense(
                self.lstm_dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)

        for dim in self.output_dims:
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)

        output = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)

        if self.critic:
            output = jnp.squeeze(output,-1)
        if self.use_sigmoid:
            output = jax.nn.sigmoid(output) 
        if self.concept_shape is not None:
            output = jnp.reshape(output,(*output.shape[:-1],*self.concept_shape))
        return hidden, output
    
    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.LSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )
    
    
