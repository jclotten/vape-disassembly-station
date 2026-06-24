import gymnasium as gym

from . import env_cfg

gym.register(
    id="Franka-PickAndPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaPickAndPlaceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:FrankaPickAndPlacePPORunnerCfg",
    },
)
