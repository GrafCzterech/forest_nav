import torch

from bridge_config import setup_logging, setup

logger = setup_logging(("forest_gen", "neuroforgelab", "main"))

simulation_app, _ = setup()

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.spot import SPOT_CFG

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotActionsCfg,
    SpotObservationsCfg,
    SpotCommandsCfg,
)

from forest_gen import ForestGenSpec


def run_manger(scene: ManagerBasedEnv):
    """Function to run a simulation.

    Args:
        scene (ManagerBasedEnv): Wrapper over the scene
    """
    while simulation_app.is_running():
        with torch.inference_mode():
            joint_efforts = torch.randn_like(scene.action_manager.action)
            scene.step(joint_efforts)


SPEC = ForestGenSpec(size=100, robot=SPOT_CFG)


@configclass
class EventCfg:

    reset_scene = SPEC.create_reset_event(scene_name="World")


# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py


@configclass
class EnvCfg(ManagerBasedEnvCfg):

    scene = SPEC.create_instance().new_scene()

    # https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html

    events = EventCfg()
    actions = SpotActionsCfg()
    observations = (
        SpotObservationsCfg()
    )  # FIXME this cfg depends on this being the RL manager, and the presence of commands

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


if __name__ == "__main__":

    cfg = EnvCfg()
    logger.debug("Config initialized")
    manager = ManagerBasedEnv(cfg)
    logger.debug("Manager initialized")
    run_manger(manager)

    simulation_app.close()
