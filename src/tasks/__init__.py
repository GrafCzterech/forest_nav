import gymnasium as gym

gym.register(
    id="forest_nav",
    entry_point="tasks.forest_nav:ForestNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forest_nav:EnvCfg",
        "skrl_cfg_entry_point": f"{__name__}:skrl_forest_nav_cfg.yaml",
    }
)
