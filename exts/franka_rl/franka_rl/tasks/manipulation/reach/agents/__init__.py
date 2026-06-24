from .rsl_rl_cfg import FrankaReachPPORunnerCfg

sb3_cfg = {
    "policy": "MlpPolicy",
    "n_steps": 24,
    "batch_size": 1024,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 8,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,
}
