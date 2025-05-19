from dataclasses import dataclass
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
    TerminationTermCfg,
    RewardTermCfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

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
    scene.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            joint_efforts = torch.randn_like(scene.action_manager.action)
            scene.step(joint_efforts)


SIZE = 256


@configclass
class EventCfg:

    def __init__(self, spec: ForestGenSpec):
        self.reset_scene = spec.create_reset_event(scene_name="World")


# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py


@dataclass
class DistanceToTargetCfg:
    """Distance to target."""

    target: torch.Tensor

    def __call__(self, cfg: ManagerBasedRLEnv) -> torch.Tensor:
        asset: RigidObject = cfg.scene["robot"]
        return torch.linalg.norm(asset.data.root_pos_w - self.target, dim=-1)


@configclass
class RewardsCfg:
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)
    terminating = RewardTermCfg(func=mdp.is_terminated, weight=-2.0)
    distance = RewardTermCfg(
        func=DistanceToTargetCfg(target=torch.tensor([0.0, 0.0, 0.0])), weight=1.0
    )


def terrain_out_of_bounds(
    cfg: ManagerBasedRLEnv, distance_buffer: float = 3.0
) -> torch.Tensor:
    asset: RigidObject = cfg.scene["robot"]
    x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > SIZE - distance_buffer
    y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > SIZE - distance_buffer
    return torch.logical_or(x_out_of_bounds, y_out_of_bounds)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    terrain_out_of_bounds = TerminationTermCfg(
        func=terrain_out_of_bounds,
        params={"distance_buffer": 3.0},
        time_out=True,
    )


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

    def __init__(self, size: int = 256):
        super().__init__()

        spec = ForestGenSpec(
            size=size,
            robot=SPOT_CFG,
        )

        self.scene = spec.create_instance().new_scene()

        # https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html

        self.episode_length_s = 100.0

        self.actions = SpotActionsCfg()  # spot effectors
        self.observations = ObservationsCfg()  # spot observations
        self.events = EventCfg(spec)  # events that happen in the scene

        # RL config
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()

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

    cfg = EnvCfg(SIZE)
    logger.debug("Config initialized")
    manager = ManagerBasedRLEnv(cfg)
    logger.debug("Manager initialized")
    run_manger(manager)

    simulation_app.close()
