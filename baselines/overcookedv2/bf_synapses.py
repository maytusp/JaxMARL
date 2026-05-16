from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import traverse_util
from flax.core import freeze, unfreeze


BF_DEFAULTS = {
    "enabled": False,
    "num_states": 4,
    "apply_to": "actor_shared",
    "exclude_critic": True,
    "exclude_norm": True,
    "dt": 1.0,
    "c_base": 1.0,
    "c_growth": 2.0,
    "tau_min": 100.0,
    "tau_max": 100000.0,
    "leak_final": True,
    "flow_every_optimizer_step": True,
    "flow_steps": 1,
    "debug_metrics": True,
}


def with_bf_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow config copy with BF defaults filled in."""
    config = dict(config)
    bf_config = dict(BF_DEFAULTS)
    bf_config.update(config.get("bf", {}) or {})
    config["bf"] = bf_config
    return config


def build_bf_constants(config: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    """Construct BF capacities, timescales, and conductances."""
    num_states = int(config["num_states"])
    if num_states < 2:
        raise ValueError("BF requires num_states >= 2")

    idx = jnp.arange(num_states, dtype=jnp.float32)
    tau_min = jnp.asarray(config["tau_min"], dtype=jnp.float32)
    tau_max = jnp.asarray(config["tau_max"], dtype=jnp.float32)
    ratio = (tau_max / tau_min) ** (1.0 / max(num_states - 1, 1))
    tau = tau_min * ratio**idx
    capacities = jnp.asarray(config["c_base"], dtype=jnp.float32) * (
        jnp.asarray(config["c_growth"], dtype=jnp.float32) ** idx
    )
    conductances = capacities[:-1] / tau[:-1]
    g_leak = capacities[-1] / tau[-1]
    stability = jnp.max(jnp.asarray(config["dt"], dtype=jnp.float32) * conductances / capacities[:-1])

    return {
        "tau": tau,
        "capacities": capacities,
        "conductances": conductances,
        "g_leak": g_leak,
        "max_dt_g_over_c": stability,
    }


def _path_string(path: Tuple[Any, ...]) -> str:
    return "/".join(str(part) for part in path)


def _has_any(path: str, needles) -> bool:
    path = path.lower()
    return any(needle in path for needle in needles)


def _has_module(path: str, module_names) -> bool:
    parts = {part.lower() for part in path.split("/")}
    return any(name.lower() in parts for name in module_names)


def _top_param_module(path: str) -> str:
    parts = path.split("/")
    if parts and parts[0] == "params":
        parts = parts[1:]
    return parts[0] if parts else ""


def _is_floating_leaf(leaf: Any) -> bool:
    return hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)


def _include_leaf(path: str, leaf: Any, config: Dict[str, Any]) -> bool:
    if not _is_floating_leaf(leaf):
        return False

    critic_names = ("critic", "value", "vf", "v_head", "value_head")
    norm_names = ("layernorm", "layer_norm", "batchnorm", "batch_norm", "norm")
    actor_names = ("actor", "policy", "pi", "actor_head", "policy_head")
    shared_names = (
        "encoder",
        "cnn",
        "mlp",
        "rnn",
        "gru",
        "lstm",
        "torso",
        "backbone",
    )

    # train_bf.py currently uses unnamed Flax modules. These fallbacks map its
    # params: CNN_0 and ScannedRNN_0 are shared, Dense_0/1 actor, Dense_2/3 value.
    unnamed_actor_modules = ("Dense_0", "Dense_1")
    unnamed_critic_modules = ("Dense_2", "Dense_3")
    unnamed_shared_modules = ("CNN_0", "ScannedRNN_0")
    unnamed_norm_modules = ("LayerNorm_0",)
    top_module = _top_param_module(path)

    custom_excludes = tuple(config.get("custom_exclude", ()) or ())
    custom_includes = tuple(config.get("custom_include", ()) or ())

    if config.get("exclude_norm", True) and (
        _has_any(path, norm_names) or top_module in unnamed_norm_modules
    ):
        return False
    if config.get("exclude_critic", True) and (
        _has_any(path, critic_names) or top_module in unnamed_critic_modules
    ):
        return False
    if custom_excludes and _has_any(path, custom_excludes):
        return False

    apply_to = config.get("apply_to", "actor_shared")
    if apply_to == "all":
        return True
    if apply_to == "actor_only":
        return _has_any(path, actor_names) or top_module in unnamed_actor_modules
    if apply_to == "actor_shared":
        is_actor = _has_any(path, actor_names) or top_module in unnamed_actor_modules
        is_shared = _has_any(path, shared_names) or top_module in unnamed_shared_modules
        return is_actor or is_shared
    if apply_to == "custom":
        return bool(custom_includes) and _has_any(path, custom_includes)

    raise ValueError(f"Unknown bf.apply_to={apply_to!r}")


def create_bf_mask(params: Any, config: Dict[str, Any]) -> Any:
    """Create a bool pytree selecting trainable leaves for BF consolidation."""
    flat_params = traverse_util.flatten_dict(unfreeze(params), keep_empty_nodes=True)
    flat_mask = {
        path: _include_leaf(_path_string(path), leaf, config)
        for path, leaf in flat_params.items()
    }
    return freeze(traverse_util.unflatten_dict(flat_mask))


def summarize_bf_mask(params: Any, bf_mask: Any) -> Dict[str, Any]:
    flat_params = traverse_util.flatten_dict(unfreeze(params), keep_empty_nodes=True)
    flat_mask = traverse_util.flatten_dict(unfreeze(bf_mask), keep_empty_nodes=True)

    selected = []
    excluded = []
    num_scalars = 0
    for path, leaf in flat_params.items():
        path_str = _path_string(path)
        is_selected = bool(flat_mask[path])
        if is_selected:
            selected.append(path_str)
            num_scalars += int(leaf.size)
        else:
            excluded.append(path_str)

    return {
        "selected_paths": selected,
        "excluded_paths": excluded,
        "num_selected_leaves": len(selected),
        "num_selected_scalars": num_scalars,
    }


def init_bf_state(params: Any, bf_mask: Any, config: Dict[str, Any]) -> Any:
    """Initialize every BF chain state to the current visible parameter."""
    del bf_mask
    num_states = int(config["num_states"])
    return jax.tree_util.tree_map(
        lambda p: jnp.stack([p] * num_states, axis=0) if _is_floating_leaf(p) else p,
        params,
    )


def apply_bf_flow_to_leaf(u: jnp.ndarray, constants: Dict[str, jnp.ndarray], config: Dict[str, Any]) -> jnp.ndarray:
    """Euler step for the Benna-Fusi chain of a single parameter tensor."""
    capacities = constants["capacities"].astype(u.dtype)
    conductances = constants["conductances"].astype(u.dtype)
    dt = jnp.asarray(config["dt"], dtype=u.dtype)

    def one_step(state):
        reshape = (conductances.shape[0],) + (1,) * (state.ndim - 1)
        g = conductances.reshape(reshape)
        c = capacities.reshape((capacities.shape[0],) + (1,) * (state.ndim - 1))
        du0_local = g[0] * (state[1] - state[0]) / c[0]
        if state.shape[0] > 2:
            middle = (
                g[:-1] * (state[:-2] - state[1:-1])
                + g[1:] * (state[2:] - state[1:-1])
            ) / c[1:-1]
            last = g[-1] * (state[-2] - state[-1]) / c[-1]
            du = jnp.concatenate([du0_local[jnp.newaxis], middle, last[jnp.newaxis]], axis=0)
        else:
            last = g[-1] * (state[-2] - state[-1]) / c[-1]
            du = jnp.stack([du0_local, last], axis=0)
        if config.get("leak_final", True):
            leak = constants["g_leak"].astype(u.dtype) * (0.0 - state[-1]) / capacities[-1].astype(u.dtype)
            du = du.at[-1].add(leak)
        return state + dt * du

    flow_steps = int(config.get("flow_steps", 1))
    return jax.lax.fori_loop(0, flow_steps, lambda _, state: one_step(state), u)


def apply_bf_flow_to_tree(
    bf_state: Any,
    bf_mask: Any,
    constants: Dict[str, jnp.ndarray],
    config: Dict[str, Any],
) -> Any:
    return jax.tree_util.tree_map(
        lambda u, m: jax.lax.cond(
            m,
            lambda x: apply_bf_flow_to_leaf(x, constants, config),
            lambda x: x,
            u,
        )
        if hasattr(u, "dtype") and jnp.issubdtype(u.dtype, jnp.floating)
        else u,
        bf_state,
        bf_mask,
    )


def _zeros_metrics(dtype=jnp.float32) -> Dict[str, jnp.ndarray]:
    return {
        "bf/selected_leaves": jnp.asarray(0, dtype=dtype),
        "bf/selected_scalars": jnp.asarray(0, dtype=dtype),
        "bf/mean_abs_u1_u2": jnp.asarray(0.0, dtype=dtype),
        "bf/mean_abs_u1_uN": jnp.asarray(0.0, dtype=dtype),
        "bf/rms_correction": jnp.asarray(0.0, dtype=dtype),
        "bf/rms_param": jnp.asarray(0.0, dtype=dtype),
        "bf/correction_to_param_ratio": jnp.asarray(0.0, dtype=dtype),
        "bf/max_dt_g_over_c": jnp.asarray(0.0, dtype=dtype),
        "bf/has_nonfinite": jnp.asarray(0.0, dtype=dtype),
    }


def _bf_metrics(
    bf_before: Any,
    bf_after: Any,
    bf_mask: Any,
    params_after_flow: Any,
    constants: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda m: jnp.asarray(m, dtype=jnp.float32), bf_mask)
    )
    selected_leaves = sum(leaves)

    def selected_sum(fn):
        vals = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(
                lambda before, after, mask, param: jnp.where(mask, fn(before, after, param), 0.0),
                bf_before,
                bf_after,
                bf_mask,
                params_after_flow,
            )
        )
        return sum(vals)

    selected_scalars = selected_sum(lambda before, after, param: jnp.asarray(param.size, dtype=jnp.float32))
    abs_u1_u2 = selected_sum(lambda before, after, param: jnp.sum(jnp.abs(after[0] - after[1])))
    abs_u1_uN = selected_sum(lambda before, after, param: jnp.sum(jnp.abs(after[0] - after[-1])))
    correction_sq = selected_sum(lambda before, after, param: jnp.sum(jnp.square(after[0] - before[0])))
    param_sq = selected_sum(lambda before, after, param: jnp.sum(jnp.square(param)))
    nonfinite = selected_sum(
        lambda before, after, param: jnp.asarray(
            jnp.logical_not(jnp.all(jnp.isfinite(after))), dtype=jnp.float32
        )
    )

    denom = jnp.maximum(selected_scalars, 1.0)
    rms_correction = jnp.sqrt(correction_sq / denom)
    rms_param = jnp.sqrt(param_sq / denom)

    return {
        "bf/selected_leaves": selected_leaves,
        "bf/selected_scalars": selected_scalars,
        "bf/mean_abs_u1_u2": abs_u1_u2 / denom,
        "bf/mean_abs_u1_uN": abs_u1_uN / denom,
        "bf/rms_correction": rms_correction,
        "bf/rms_param": rms_param,
        "bf/correction_to_param_ratio": rms_correction / (rms_param + 1e-12),
        "bf/max_dt_g_over_c": constants["max_dt_g_over_c"],
        "bf/has_nonfinite": jnp.where(nonfinite > 0, 1.0, 0.0),
    }


def bf_after_optimizer_update(
    params_after_optimizer: Any,
    bf_state: Any,
    bf_mask: Any,
    constants: Dict[str, jnp.ndarray],
    config: Dict[str, Any],
) -> Tuple[Any, Any, Dict[str, jnp.ndarray]]:
    """Write Optax-visible params into u1, flow BF chains, then expose new u1."""
    bf_before_flow = jax.tree_util.tree_map(
        lambda u, p, m: jax.lax.cond(m, lambda _: u.at[0].set(p), lambda _: u, operand=None)
        if hasattr(u, "dtype") and jnp.issubdtype(u.dtype, jnp.floating)
        else u,
        bf_state,
        params_after_optimizer,
        bf_mask,
    )

    if config.get("flow_every_optimizer_step", True):
        bf_after_flow = apply_bf_flow_to_tree(bf_before_flow, bf_mask, constants, config)
    else:
        bf_after_flow = bf_before_flow

    new_params = jax.tree_util.tree_map(
        lambda p, u, m: jax.lax.cond(m, lambda _: u[0], lambda _: p, operand=None)
        if hasattr(p, "dtype") and jnp.issubdtype(p.dtype, jnp.floating)
        else p,
        params_after_optimizer,
        bf_after_flow,
        bf_mask,
    )

    if config.get("debug_metrics", True):
        metrics = _bf_metrics(bf_before_flow, bf_after_flow, bf_mask, new_params, constants)
    else:
        metrics = _zeros_metrics()
    return new_params, bf_after_flow, metrics


def bf_disabled_metrics(constants: Dict[str, jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
    metrics = _zeros_metrics()
    if constants is not None:
        metrics["bf/max_dt_g_over_c"] = constants["max_dt_g_over_c"]
    return metrics
