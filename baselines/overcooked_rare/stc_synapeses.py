from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import traverse_util
from flax.core import freeze, unfreeze


STC_DEFAULTS = {
    "enable_stc": True,
    "theta_tag": 1e-3,
    "tag_mode": "soft",
    "tag_temperature": 1e-3,
    "eta_slow": 1e-5,
    "top_k": None,
    "top_k_fraction": 0.1,
    "capture_norm": "none",
    "capture_clip_max": None,
    "stc_apply_to": "actor_only",
    "stc_exclude_norm": True,
    "latent_dim": 64,
    "latent_lr": 1e-3,
}


def with_stc_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow config copy with top-level STC defaults filled in."""
    config = dict(config)
    for key, value in STC_DEFAULTS.items():
        config.setdefault(key, value)
    return config


def _path_string(path: Tuple[Any, ...]) -> str:
    return "/".join(str(part) for part in path)


def _has_any(path: str, needles) -> bool:
    path = path.lower()
    return any(needle in path for needle in needles)


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
    norm_names = ("layernorm", "layer_norm", "batchnorm", "batch_norm", "norm")

    # train_ppo.py uses unnamed Flax modules: CNN_0 and ScannedRNN_0 are shared,
    # Dense_0/1 are actor, and Dense_2/3 are critic.
    unnamed_actor_modules = ("Dense_0", "Dense_1")
    unnamed_shared_modules = ("CNN_0", "ScannedRNN_0")
    unnamed_norm_modules = ("LayerNorm_0",)
    top_module = _top_param_module(path)

    if config.get("stc_exclude_norm", True) and (
        _has_any(path, norm_names) or top_module in unnamed_norm_modules
    ):
        return False

    apply_to = config.get("stc_apply_to", "actor_only")
    if apply_to == "actor_only":
        return _has_any(path, actor_names) or top_module in unnamed_actor_modules
    if apply_to == "actor_shared":
        is_actor = _has_any(path, actor_names) or top_module in unnamed_actor_modules
        is_shared = _has_any(path, shared_names) or top_module in unnamed_shared_modules
        return is_actor or is_shared
    if apply_to == "all":
        return True

    raise ValueError(f"Unknown stc_apply_to={apply_to!r}")


def create_stc_mask(params: Any, config: Dict[str, Any]) -> Any:
    flat_params = traverse_util.flatten_dict(unfreeze(params), keep_empty_nodes=True)
    flat_mask = {
        path: _include_leaf(_path_string(path), leaf, config)
        for path, leaf in flat_params.items()
    }
    return freeze(traverse_util.unflatten_dict(flat_mask))


def summarize_stc_mask(params: Any, stc_mask: Any) -> Dict[str, Any]:
    flat_params = traverse_util.flatten_dict(unfreeze(params), keep_empty_nodes=True)
    flat_mask = traverse_util.flatten_dict(unfreeze(stc_mask), keep_empty_nodes=True)
    selected = []
    excluded = []
    num_scalars = 0
    for path, leaf in flat_params.items():
        path_str = _path_string(path)
        if bool(flat_mask[path]):
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


def topkmean(errors: jnp.ndarray, config: Dict[str, Any]) -> jnp.ndarray:
    traj_len = errors.shape[0]
    if config.get("top_k", None) is not None:
        k = int(config["top_k"])
    else:
        frac = config.get("top_k_fraction", None)
        k = int(frac * traj_len) if frac is not None else traj_len
    k = max(1, min(k, traj_len))
    values, _ = jax.lax.top_k(jnp.swapaxes(errors, 0, -1), k)
    return values.mean(axis=-1)


def compute_tags(eligibility: Any, stc_mask: Any, config: Dict[str, Any]) -> Any:
    theta_tag = jnp.asarray(config["theta_tag"], dtype=jnp.float32)
    temperature = jnp.maximum(jnp.asarray(config["tag_temperature"], dtype=jnp.float32), 1e-8)
    tag_mode = config.get("tag_mode", "soft")

    def tag_leaf(e, mask):
        if not _is_floating_leaf(e):
            return e
        if tag_mode == "hard":
            tag = (jnp.abs(e) > theta_tag).astype(e.dtype)
        elif tag_mode == "soft":
            tag = jax.nn.sigmoid((jnp.abs(e) - theta_tag) / temperature).astype(e.dtype)
        else:
            raise ValueError(f"Unknown tag_mode={tag_mode!r}")
        return jnp.where(mask, tag, jnp.zeros_like(tag))

    return jax.tree_util.tree_map(tag_leaf, eligibility, stc_mask)


def normalize_capture(
    capture_raw: jnp.ndarray,
    running_mean: jnp.ndarray,
    running_var: jnp.ndarray,
    running_count: jnp.ndarray,
    config: Dict[str, Any],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if config.get("capture_norm", "none") == "running_zscore":
        batch_mean = capture_raw.mean()
        batch_var = capture_raw.var()
        count = running_count + 1.0
        delta = batch_mean - running_mean
        new_mean = running_mean + delta / count
        new_var = running_var + (batch_var - running_var) / count
        capture = (capture_raw - new_mean) / (jnp.sqrt(new_var) + 1e-8)
    elif config.get("capture_norm", "none") == "none":
        new_mean = running_mean
        new_var = running_var
        count = running_count
        capture = capture_raw
    else:
        raise ValueError(f"Unknown capture_norm={config.get('capture_norm')!r}")

    clip_max = config.get("capture_clip_max", None)
    if clip_max is not None:
        capture = jnp.clip(capture, a_min=None, a_max=float(clip_max))
    return capture, new_mean, new_var, count


def stc_disabled_metrics(dtype=jnp.float32) -> Dict[str, jnp.ndarray]:
    return {
        "stc/tag_density": jnp.asarray(0.0, dtype=dtype),
        "stc/eligibility_norm": jnp.asarray(0.0, dtype=dtype),
        "stc/slow_update_norm": jnp.asarray(0.0, dtype=dtype),
        "stc/capture": jnp.asarray(0.0, dtype=dtype),
        "stc/capture_surprise": jnp.asarray(0.0, dtype=dtype),
        "stc/latent_pred_error": jnp.asarray(0.0, dtype=dtype),
        "stc/latent_pred_error_rare": jnp.asarray(0.0, dtype=dtype),
        "stc/latent_pred_error_common": jnp.asarray(0.0, dtype=dtype),
    }
