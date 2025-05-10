import torch

from bridge_config import setup_logging, setup

logger = setup_logging(("forest_gen", "neuroforgelab", "main"))

simulation_app, _ = setup()

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationTermCfg,
    SceneEntityCfg,
    ObservationGroupCfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# isaaclab_assets is a special module containing prebuilt assets
from isaaclab_assets.robots.spot import SPOT_CFG

# isaaclab_tasks is a special module containing prebuilt training tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotActionsCfg,
)
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from forest_gen import ForestGenSpec


def run_manger(scene: ManagerBasedRLEnv):
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
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    pass


@configclass
class CommandsCfg:
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):

    scene = SPEC.create_instance().new_scene()

    # https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html

    episode_length_s = 100.0

    actions = SpotActionsCfg()  # spot effectors
    observations = ObservationsCfg()  # spot observations
    events = EventCfg()  # events that happen in the scene

    # RL config
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

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
    manager = ManagerBasedRLEnv(cfg)
    logger.debug("Manager initialized")
    run_manger(manager)

    simulation_app.close()
