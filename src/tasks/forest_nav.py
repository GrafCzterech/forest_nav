import torch
from collections.abc import Sequence

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ViewerCfg
from isaaclab.envs.mdp.recorders import (
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.managers import (
    EventTermCfg as EventTerm,
)
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
)
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RecorderManagerBaseCfg,
    DatasetExportMode,
    SceneEntityCfg,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# isaaclab_assets is a special module containing prebuilt assets
from isaaclab_assets.robots.spot import SPOT_CFG

# isaaclab_tasks is a special module containing prebuilt training tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotActionsCfg,
    SpotCommandsCfg,
)

# our code
from stripe_kit import TaskEnvCfg, TrainingSpec
from bridge_config import get_asset_path, setup_logging
from forest_gen import ForestGenSpec

logger = setup_logging(("forest_gen", "neuroforgelab", "main"))

SIZE = 50
"""The size of the scene. The generated scene will be a square of size SIZE x SIZE."""


def ensure_above_ground(
    env: TaskEnvCfg, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> None:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.root_state_w[env_ids].clone()
    positions = root_states[:, 0:3]

    for i in range(positions.shape[0]):
        try:
            ground_height = env.terrain.raw(
                positions[i][0].cpu().item(), positions[i][1].cpu().item()
            )
        except IndexError:
            ground_height = 0.0

        positions[i][2] = max(
            ground_height + 0.5,
            positions[i][2].cpu().item(),
        )

    asset.write_root_pose_to_sim(
        torch.cat([positions, root_states[:, 3:7]], dim=-1),
        env_ids=env_ids,  # pyright: ignore[reportArgumentType]
    )


# No, you cannot have both event recorder handler things in the same class,
# because that would make too much sense


class StepPositionsRecorder(RecorderTerm):
    def record_pre_step(
        self, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        asset: Articulation = self._env.scene[asset_cfg.name]
        positions = asset.data.root_state_w[:, 0:3]
        assert asset.num_instances == len(positions)
        return "root_state_pre", {
            "positions": positions,
            "headings": asset.data.heading_w,
        }

    def record_post_step(
        self, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        asset: Articulation = self._env.scene[asset_cfg.name]
        positions = asset.data.root_state_w[:, 0:3]
        assert asset.num_instances == len(positions)
        return "root_state_post", {
            "positions": positions,
            "headings": asset.data.heading_w,
        }


@configclass
class RecordersCfg(RecorderManagerBaseCfg):
    """Recorders terms for the MDP."""

    dataset_export_dir_path = "logs/dataset"
    dataset_export_mode = DatasetExportMode.EXPORT_ALL
    step_states = RecorderTermCfg(StepPositionsRecorder)


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
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
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

    """
    push_by_setting_velocity = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    """

    ensure_above_ground = EventTerm(
        func=ensure_above_ground,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py


def traversability(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    device = asset.data.root_pos_w.device

    positions = asset.data.root_pos_w  # [num_envs, 3]
    x_indices = torch.clamp(
        positions[:, 0].int(),
        min=0,
        max=env.terrain.traversability_score.shape[0] - 1,
    )
    y_indices = torch.clamp(
        positions[:, 1].int(),
        min=0,
        max=env.terrain.traversability_score.shape[1] - 1,
    )

    score_tensor = torch.tensor(env.terrain.traversability_score, device=device)
    traversability_scores = score_tensor[x_indices, y_indices]  # [num_envs]
    travers = torch.where(
        traversability_scores < 0.3,
        -torch.ones_like(traversability_scores),
        torch.ones_like(traversability_scores),
    )

    return travers


def distance_to_target(
    env, asset_cfg: SceneEntityCfg, target: torch.Tensor | None = None
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    device = asset.data.root_pos_w.device
    # use only x,y coordinates for distance calculation
    pos_xy = asset.data.root_pos_w[..., :2]
    if target is None:
        target_xy = torch.tensor([0.0, 0.0], device=device)
    elif not isinstance(target, torch.Tensor):
        target_xy = torch.tensor(target, device=device)[..., :2]
    else:
        target_xy = target.to(device)[..., :2]

    return torch.linalg.norm(pos_xy - target_xy, dim=-1)


# configclass to setup the reward terms, this is used directly in the RL learning process
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.RewardTermCfg
@configclass
class RewardsCfg:
    # sanity
    alive = RewTerm(func=mdp.is_alive, weight=3)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # rewards
    traversability = RewTerm(
        func=traversability,
        weight=7.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    """
    distance = RewTerm(
        func=distance_to_target,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target": [0.0, 0.0]},
    )
    """

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

    gait = RewTerm(
        func=spot_mdp.GaitReward,
        weight=10.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (
                ("fl_foot", "hr_foot"),
                ("fr_foot", "hl_foot"),
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

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
                "contact_forces",
                body_names=["body", ".*leg"],  # ".*leg"
            ),
            "threshold": 1.0,
        },
    )
    # joints_limit = DoneTerm(
    #     func=mdp.joint_pos_out_of_limit,
    # )


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


terrain_spec = ForestGenSpec(asset_path=get_asset_path(), size=SIZE, margin=20)

task_spec = TrainingSpec(
    scene=terrain_spec,
    robot=SPOT_CFG.copy(),
    actions=SpotActionsCfg(),
    observations=ObservationsCfg(),
    events=EventCfg(),
    rewards=RewardsCfg(),
    terminations=TerminationsCfg(),
    commands=SpotCommandsCfg(),
    recorders=RecordersCfg(),
    sensors={
        "contact_forces": ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/.*",
            history_length=3,
            track_air_time=True,
            filter_prim_paths_expr=[
                "/terrain/terrain_0/mesh",
                "/terrain/terrain_1/mesh",
            ],
        )
    },
).to_env_cfg(
    ViewerCfg(
        eye=(3, 3, 3),
        origin_type="asset_root",
        asset_name="robot",
    ),
    4,
    10,
)
