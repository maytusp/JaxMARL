import jax
import jax.numpy as jnp

from baselines.foraging.train_lmpred import (
    ForagingPerspectiveTransform,
    ScannedRNN,
    TwoStreamForagingActorCriticRNN,
)


def test_foraging_perspective_transform_swaps_both_partners():
    obs = jnp.arange(18, dtype=jnp.float32)

    partner_views = ForagingPerspectiveTransform(
        num_agents=3,
        num_resources=2,
        capability_dim=1,
    )(obs)

    expected_partner_1 = jnp.array(
        [
            3,
            4,
            5,
            0,
            1,
            2,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            15,
            17,
        ],
        dtype=jnp.float32,
    )
    expected_partner_2 = jnp.array(
        [
            6,
            7,
            8,
            3,
            4,
            5,
            0,
            1,
            2,
            9,
            10,
            11,
            12,
            13,
            14,
            17,
            16,
            15,
        ],
        dtype=jnp.float32,
    )

    assert partner_views.shape == (2, 18)
    assert jnp.all(partner_views[0] == expected_partner_1)
    assert jnp.all(partner_views[1] == expected_partner_2)


def test_foraging_lmpred_network_initializes_on_flat_obs():
    config = {
        "ENV_KWARGS": {
            "num_agents": 3,
            "num_resources": 2,
        },
        "ACTIVATION": "relu",
        "ENCODER_HIDDEN_DIM": 32,
        "ENCODER_NUM_LAYERS": 2,
        "FC_DIM_SIZE": 32,
        "GRU_HIDDEN_DIM": 32,
        "CAPABILITY_DIM": 1,
        "PERSPECTIVE_TRANSFORM": True,
        "FINETUNE_SELF_STREAM": True,
        "FINETUNE_OTHER_STREAM": False,
        "SELF_PRED_GAMMAS": (0.0, 0.5, 0.9),
    }
    network = TwoStreamForagingActorCriticRNN(action_dim=5, config=config)
    obs = jnp.ones((1, 6, 18), dtype=jnp.float32)
    dones = jnp.zeros((1, 6), dtype=bool)
    hidden = ScannedRNN.initialize_carry(6, 3 * config["GRU_HIDDEN_DIM"])

    params = network.init(jax.random.PRNGKey(0), hidden, (obs, dones))
    next_hidden, pi, value, aux = network.apply(params, hidden, (obs, dones))

    assert next_hidden.shape == (6, 96)
    assert pi.logits.shape == (1, 6, 5)
    assert value.shape == (1, 6)
    assert aux["rnn_hidden"].shape == (1, 6, 96)
    assert aux["pred_hidden_repr"].shape == (1, 6, 3, 96)
