import jax.numpy as jnp
from flax.core import freeze

from bf_synapses import (
    apply_bf_flow_to_leaf,
    bf_after_optimizer_update,
    build_bf_constants,
    create_bf_mask,
    init_bf_state,
    with_bf_defaults,
)


def _cfg(**overrides):
    return with_bf_defaults({"bf": {"enabled": True, **overrides}})["bf"]


def _params():
    return freeze(
        {
            "params": {
                "CNN_0": {"kernel": jnp.ones((2, 2))},
                "LayerNorm_0": {"scale": jnp.ones((2,))},
                "Dense_0": {"bias": jnp.ones((2,))},
                "Dense_2": {"bias": jnp.ones((1,))},
            }
        }
    )


def test_default_mask_selects_shared_and_actor_but_not_norm_or_critic():
    mask = create_bf_mask(_params(), _cfg(apply_to="actor_shared"))

    assert bool(mask["params"]["CNN_0"]["kernel"])
    assert bool(mask["params"]["Dense_0"]["bias"])
    assert not bool(mask["params"]["Dense_2"]["bias"])
    assert not bool(mask["params"]["LayerNorm_0"]["scale"])


def test_actor_only_mask_excludes_shared_and_critic():
    mask = create_bf_mask(_params(), _cfg(apply_to="actor_only"))

    assert not bool(mask["params"]["CNN_0"]["kernel"])
    assert bool(mask["params"]["Dense_0"]["bias"])
    assert not bool(mask["params"]["Dense_2"]["bias"])


def test_init_and_equal_chain_flow_are_identity():
    config = _cfg()
    params = _params()
    mask = create_bf_mask(params, config)
    state = init_bf_state(params, mask, config)
    constants = build_bf_constants(config)

    assert jnp.allclose(state["params"]["CNN_0"]["kernel"][0], params["params"]["CNN_0"]["kernel"])
    assert jnp.allclose(apply_bf_flow_to_leaf(jnp.ones((config["num_states"], 3)), constants, config), 1.0)


def test_visible_state_moves_toward_hidden_state():
    config = _cfg(leak_final=False)
    constants = build_bf_constants(config)
    u = jnp.ones((config["num_states"], 3))
    u = u.at[0].set(2.0)

    flowed = apply_bf_flow_to_leaf(u, constants, config)

    assert jnp.all(flowed[0] < u[0])
    assert jnp.all(flowed[0] > u[1])


def test_after_optimizer_update_preserves_tree_and_metrics_are_finite():
    config = _cfg()
    params = _params()
    mask = create_bf_mask(params, config)
    state = init_bf_state(params, mask, config)
    constants = build_bf_constants(config)

    new_params, new_state, metrics = bf_after_optimizer_update(params, state, mask, constants, config)

    assert new_params.keys() == params.keys()
    assert new_state.keys() == state.keys()
    assert jnp.isfinite(metrics["bf/rms_correction"])
