import gymnasium as gym

from . import env_cfg

gym.register(
    id="Franka-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:FrankaReachPPORunnerCfg",
        "sb3_cfg_entry_point": f"{__name__}.agents:sb3_cfg",
    },
)

gym.register(
    id="Franka-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaReachEnvCfg_Play,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:FrankaReachPPORunnerCfg",
        "sb3_cfg_entry_point": f"{__name__}.agents:sb3_cfg",
    },
)
