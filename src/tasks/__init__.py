from .forest_nav import task_spec

task_spec.register(
    "forest_nav", skrl_cfg_entry_point=f"{__name__}:skrl_forest_nav_cfg.yaml"
)
