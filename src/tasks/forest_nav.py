from dataclasses import dataclass
import torch

from bridge_config import setup_logging, setup

logger = setup_logging(("forest_gen", "neuroforgelab", "main"))

# simulation_app, _ = setup()

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
    ObservationGroupCfg as ObsGroup,
    TerminationTermCfg as DoneTerm,
    RewardTermCfg as RewTerm,
    EventTermCfg as EventTerm,
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
import isaaclab_tasks.manager_based.navigation.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp

from forest_gen import ForestGenSpec


SIZE = 256
"""The size of the scene. The generated scene will be a square of size SIZE x SIZE."""


def reset_robot_pos(cfg: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset the robot position to the center of the scene.

    Args:
        cfg (ManagerBasedRLEnv): The configuration of the scene.
    """
    print("test", env_ids)
    asset: RigidObject = cfg.scene["robot"]
    asset.data.root_pos_w[:, 0] = SIZE / 2.0
    asset.data.root_pos_w[:, 1] = SIZE / 2.0
    asset.data.root_pos_w[:, 2] = 0.0


# configclass to setup events that the manager will invoke at certain times
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.EventTermCfg
@configclass
class EventCfg:

    def __init__(self):
        self.reset_scene = EventTerm(reset_robot_pos, mode="reset", params={})


# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py


def distance_to_target(
    cfg: ManagerBasedRLEnv, target: torch.Tensor | None = None
) -> torch.Tensor:
    asset: RigidObject = cfg.scene["robot"]
    device = asset.data.root_pos_w.device
    if target is None:
        target = torch.tensor([0.0, 0.0, 0.0], device=device)
    elif not isinstance(target, torch.Tensor):
        target = torch.tensor(target, device=device)
    return torch.linalg.norm(asset.data.root_pos_w - target, dim=-1)


def distance_from_start(cfg: "ForestNavEnv") -> torch.Tensor:
    asset: RigidObject = cfg.scene["robot"]
    start_point = cfg.cfg.spec.start_point  # type: ignore
    pos_xy = asset.data.root_pos_w[:, :2]
    start_xy = torch.tensor(
        start_point[:2], device=asset.data.root_pos_w.device
    )
    return torch.linalg.norm(pos_xy - start_xy, dim=-1)


# configclass to setup the reward terms, this is used directly in the RL learning process
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.RewardTermCfg
@configclass
class RewardsCfg:

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    distance = RewTerm(
        func=distance_to_target, params={"target": [0.0, 0.0, 0.0]}, weight=1.0
    )
    move = RewTerm(func=distance_from_start, weight=-0.1)
    # TODO reward for keeping joints in "normal" ranges


def terrain_out_of_bounds(
    cfg: ManagerBasedRLEnv, distance_buffer: float = 3.0
) -> torch.Tensor:
    """Check if the robot is out of bounds of the terrain.

    Args:
        cfg (ManagerBasedRLEnv): The configuration of the scene.
        distance_buffer (float, optional): The distance the agent should keep from the edge. Defaults to 3.0.

    Returns:
        torch.Tensor: _description_
    """
    asset: RigidObject = cfg.scene["robot"]
    x_out_of_bounds = (
        torch.abs(asset.data.root_pos_w[:, 0]) > SIZE - distance_buffer
    )
    y_out_of_bounds = (
        torch.abs(asset.data.root_pos_w[:, 1]) > SIZE - distance_buffer
    )
    return torch.logical_or(x_out_of_bounds, y_out_of_bounds)


# configclass to setup the termination terms, if any of these return true, the episode is terminated
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.TerminationTermCfg
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=terrain_out_of_bounds,
        params={"distance_buffer": 3.0},
        time_out=True,
    )
    # TODO kill when joints would die


# configclass to setup the observations, these provide arguments for the RL model
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationTermCfg
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # this is straight copied from the tutorial, probably works ok?

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# one configclass to rule them all and in the darkness bind them
@configclass
class EnvCfg(ManagerBasedRLEnvCfg):

    def __init__(self, size: int = 100):
        self.size = size
        self.spec = ForestGenSpec(size=self.size, robot=SPOT_CFG)
        super().__init__()

        self.robot = SPOT_CFG
        self.robot.prim_path = "/World/terrain/robot"

        self.scene = InteractiveSceneCfg()
        setattr(self.scene, "robot", SPOT_CFG)

        self.scene.env_spacing = 1.0
        self.scene.num_envs = 1

        # https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html

        self.episode_length_s = 100.0

        self.actions = SpotActionsCfg()  # spot effectors
        self.observations = ObservationsCfg()  # spot observations
        self.events = EventCfg()  # events that happen in the scene

        # RL config
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()

    # i have no clue why this is needed, but it was in the tutorial and I am a pious man who follows the path of the righteous
    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        # self.viewer.eye = (50.0, 50.0, 200.0)
        # self.viewer.lookat = (50.0, 50.0, 0.0)
        self.viewer.eye = (
            self.spec.start_point[0] - 10,
            self.spec.start_point[1] - 10,
            10,
        )
        self.viewer.lookat = (
            self.spec.start_point[0],
            self.spec.start_point[1],
            1,
        )
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


class ForestNavEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: EnvCfg, **kwargs):
        # Tutaj generujesz scenę na podstawie parametrów z cfg
        cfg.scene = cfg.spec.create_instance().new_scene()
        super().__init__(cfg, **kwargs)
        self.cfg = cfg
