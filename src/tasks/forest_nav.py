import torch

from bridge_config import setup_logging

logger = setup_logging(("forest_gen", "neuroforgelab", "main"))

# simulation_app, _ = setup()

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import Articulation

# isaaclab_assets is a special module containing prebuilt assets
from isaaclab_assets.robots.spot import SPOT_CFG

# isaaclab_tasks is a special module containing prebuilt training tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotActionsCfg,
    SpotCommandsCfg,
)
import isaaclab_tasks.manager_based.navigation.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp

from forest_gen import ForestGenSpec, TraversabilityMapBuilder


SIZE = 256
"""The size of the scene. The generated scene will be a square of size SIZE x SIZE."""


# configclass to setup events that the manager will invoke at certain times
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.EventTermCfg
@configclass
class EventCfg:

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )



# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py


def distance_to_asset(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_key: str) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    targets = [a for name, a in env.scene.extras.items() if target_key in name]
    if not targets:
        return torch.tensor(0.0)
    #https://docs.isaacsim.omniverse.nvidia.com/5.0.0/py/source/extensions/isaacsim.core.prims/docs/index.html#isaacsim.core.prims.XFormPrim
    return torch.Tensor(tuple(min(torch.norm(pos - target.get_world_poses()[0]) for target in targets) for pos in asset.data.root_pos_w))

def traversability(env: "ForestNavEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.Tensor(env.trav_map[asset.data.root_pos_w[0], asset.data.root_pos_w[1]])

# configclass to setup the reward terms, this is used directly in the RL learning process
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.RewardTermCfg
@configclass
class RewardsCfg:
    # sanity
    alive = RewTerm(func=mdp.is_alive, weight=3)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


    # rewards
    air_time = RewTerm(
        func=spot_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot"
            ),
        },
    )
    base_angular_velocity = RewTerm(
        func=spot_mdp.base_angular_velocity_reward,
        weight=2.5,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewTerm(
        func=spot_mdp.base_linear_velocity_reward,
        weight=10.0,
        params={
            "std": 1.0,
            "ramp_rate": 0.5,
            "ramp_at_vel": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    foot_clearance = RewTerm(
        func=spot_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    """
    traversability = RewTerm(
        func=traversability,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )"""

    # penalties
    action_smoothness = RewTerm(
        func=spot_mdp.action_smoothness_penalty, weight=-1.0
    )
    base_motion = RewTerm(
        func=spot_mdp.base_motion_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewTerm(
        func=spot_mdp.base_orientation_penalty,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot"
            ),
            "threshold": 1.0,
        },
    )
    joint_acc = RewTerm(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_vel = RewTerm(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_pos = RewTerm(
        func=spot_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewTerm(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )


# configclass to setup the termination terms, if any of these return true, the episode is terminated
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.TerminationTermCfg
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["body", ".*leg"]
            ),
            "threshold": 1.0,
        },
    )


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
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
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

    spec = ForestGenSpec(size=SIZE)

    scene = InteractiveSceneCfg(1, 1.0)
    setattr(scene, "robot", SPOT_CFG)

    # https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html

    episode_length_s = 100.0

    actions = SpotActionsCfg()  # spot effectors
    observations = ObservationsCfg()  # spot observations
    events = EventCfg()  # events that happen in the scene

    # RL config
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    commands = SpotCommandsCfg()

    viewer = ViewerCfg(
        eye=(spec.origin[0] - 5, spec.origin[1] - 5, 10),
        lookat=(spec.origin[0], spec.origin[1], 1),
        asset_name="robot",
    )

    # i have no clue why this is needed, but it was in the tutorial and I am a pious man who follows the path of the righteous
    def __post_init__(self):
        """Post initialization."""
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


class ForestNavEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: EnvCfg, **kwargs):
        # Tutaj generujesz scenę na podstawie parametrów z cfg
        factory = cfg.spec.create_instance(cfg.scene.num_envs, cfg.scene.env_spacing)
        builder: TraversabilityMapBuilder = factory.terrain.traversability_map # type: ignore
        self.trav_map = builder.get_score()
        factory.set_robot(SPOT_CFG)
        factory.add_sensor("contact_forces", ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/.*",
            history_length=3,
            track_air_time=True,
        ))
        cfg.scene = factory.get_scene()
        super().__init__(cfg, **kwargs)
        self.cfg = cfg
