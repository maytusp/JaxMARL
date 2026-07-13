METHOD_MODULES = {
    "sp": "baselines.overcookedv2.train_sp",
    "bf_sp": "baselines.overcookedv2.train_bf_sp",
    "privz": "baselines.overcookedv2.train_privz",
    "ph2_sp": "baselines.overcookedv2.train_ph2_sp",
    "e3t": "baselines.overcookedv2.train_e3t",
    "e3tlm": "baselines.overcookedv2.train_e3tlm",
    "ph2v4": "baselines.overcookedv2.train_ph2v4",
    "ph2v4_ablate": "baselines.overcookedv2.train_ph2v4",
    "ph2v5": "baselines.overcookedv2.train_ph2v5",
    "ph2v5_ablate": "baselines.overcookedv2.train_ph2v5",
    "lmpred": "baselines.overcookedv2.train_lmpred",
    "lmpred_ablate": "baselines.overcookedv2.train_lmpred",
    "lmpred_gamma0": "baselines.overcookedv2.train_lmpred",
    "lmpred_gamma0_ablate": "baselines.overcookedv2.train_lmpred",
    "lmpred_gamma09": "baselines.overcookedv2.train_lmpred",
    "lmpred_gamma09_ablate": "baselines.overcookedv2.train_lmpred",
    "lmpred_no_self_pred": "baselines.overcookedv2.train_lmpred",
    "lmpred_ablate_no_self_pred": "baselines.overcookedv2.train_lmpred",
    "lmpredlow": "baselines.overcookedv2.train_lmpredlow",
    "lmpredlow_ablate": "baselines.overcookedv2.train_lmpredlow",
    "lmpredlow_ema": "baselines.overcookedv2.train_lmpredlow_ema",
    "lmpredlow_ema_ablate": "baselines.overcookedv2.train_lmpredlow_ema",
    "lmpred_pop": "baselines.overcookedv2.train_lmpred_pop",
    "lmpred_pop_ablate": "baselines.overcookedv2.train_lmpred_pop",
    "lmpred_ema": "baselines.overcookedv2.train_lmpred_ema",
    "lmpred_ema_ablate": "baselines.overcookedv2.train_lmpred_ema",
    "lmpred_ema_gamma0": "baselines.overcookedv2.train_lmpred_ema",
    "lmpred_ema_gamma09": "baselines.overcookedv2.train_lmpred_ema",
    "lmpred_ema_no_self_pred": "baselines.overcookedv2.train_lmpred_ema",
    "fcp": "baselines.overcookedv2.train_fcp",
    "mep_pool": "baselines.overcookedv2.train_mep",
    "mep_br": "baselines.overcookedv2.train_mep",
    "pbt": "baselines.overcookedv2.train_pbt",

}

TWO_STREAM_METHODS = {
    "ph2_v1",
    "ph2_v2",
    "ph2_v2_ablate",
    "ph2_sp",
    "dual",
    "dual_ablation",
    "e3tlm",
    "ph2sf",
    "ph2sf_ablate",
    "ph2v3",
    "ph2v3_ablate",
    "ph2v4",
    "ph2v4_ablate",
    "ph2v5",
    "ph2v5_ablate",
    "lmpred",
    "lmpred_ablate",
    "lmpred_gamma0",
    "lmpred_gamma0_ablate",
    "lmpred_gamma09",
    "lmpred_gamma09_ablate",
    "lmpred_no_self_pred",
    "lmpred_ablate_no_self_pred",
    "lmpredlow",
    "lmpredlow_ablate",
    "lmpredlow_ema",
    "lmpredlow_ema_ablate",
    "lmpred_pop",
    "lmpred_pop_ablate",
    "lmpred_ema",
    "lmpred_ema_ablate",
    "lmpred_ema_gamma0",
    "lmpred_ema_gamma09",
    "lmpred_ema_no_self_pred",
}

FUSION_HIDDEN_METHODS = {
    "ph2_v2",
    "ph2_v2_ablate",
    "dual",
    "dual_ablation",
    "e3tlm",
    "ph2sf",
    "ph2sf_ablate",
    "ph2v3",
    "ph2v3_ablate",
    "ph2v4",
    "ph2v4_ablate",
    "ph2v5",
    "ph2v5_ablate",
    "lmpred",
    "lmpred_ablate",
    "lmpred_gamma0",
    "lmpred_gamma0_ablate",
    "lmpred_gamma09",
    "lmpred_gamma09_ablate",
    "lmpred_no_self_pred",
    "lmpred_ablate_no_self_pred",
    "lmpredlow",
    "lmpredlow_ablate",
    "lmpredlow_ema",
    "lmpredlow_ema_ablate",
    "lmpred_pop",
    "lmpred_pop_ablate",
    "lmpred_ema",
    "lmpred_ema_ablate",
    "lmpred_ema_gamma0",
    "lmpred_ema_gamma09",
    "lmpred_ema_no_self_pred",
}

LMPRED_EMA_METHOD_CONFIGS = {
    "lmpred": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpred_ablate": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpred_gamma0": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpred_gamma0_ablate": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpred_gamma09": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpred_gamma09_ablate": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpred_no_self_pred": {
        "SELF_PRED_COEF": 0.0,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpred_ablate_no_self_pred": {
        "SELF_PRED_COEF": 0.0,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpredlow": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpredlow_ablate": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpredlow_ema": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpredlow_ema_ablate": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpred_ema": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpred_ema_ablate": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpred_ema_gamma0": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpred_ema_gamma09": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpred_ema_no_self_pred": {
        "SELF_PRED_COEF": 0.0,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": True,
    },
}

LMPRED_V2_METHOD_CONFIGS = {
    "lmpredv2_ablate_no_self_pred": {
        "SELF_PRED_COEF": 0.0,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpredv2_no_self_pred": {
        "SELF_PRED_COEF": 0.0,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpredv202": {
        "SELF_PRED_COEF": 0.2,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpredv202_ablate": {
        "SELF_PRED_COEF": 0.2,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpredv202_gamma0": {
        "SELF_PRED_COEF": 0.2,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpredv202_gamma0_ablate": {
        "SELF_PRED_COEF": 0.2,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpredv204": {
        "SELF_PRED_COEF": 0.4,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpredv204_ablate": {
        "SELF_PRED_COEF": 0.4,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpredv204_gamma0": {
        "SELF_PRED_COEF": 0.4,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpredv204_gamma0_ablate": {
        "SELF_PRED_COEF": 0.4,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpredv2005": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpredv2005_ablate": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0, 0.5, 0.9],
        "PERSPECTIVE_TRANSFORM": False,
    },
    "lmpredv2005_gamma0": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": True,
    },
    "lmpredv2005_gamma0_ablate": {
        "SELF_PRED_COEF": 0.05,
        "SELF_PRED_GAMMAS": [0.0],
        "PERSPECTIVE_TRANSFORM": False,
    },
}

METHOD_MODULES.update(
    {
        method: "baselines.overcookedv2.train_lmpredV2"
        for method in LMPRED_V2_METHOD_CONFIGS
    }
)
TWO_STREAM_METHODS.update(LMPRED_V2_METHOD_CONFIGS)
FUSION_HIDDEN_METHODS.update(LMPRED_V2_METHOD_CONFIGS)
LMPRED_EMA_METHOD_CONFIGS.update(LMPRED_V2_METHOD_CONFIGS)
