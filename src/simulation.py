import argparse

from bridge_config import get_asset_path, setup_logging
from isaaclab.app import AppLauncher

logger = setup_logging(("forest_gen", "neuroforgelab", "main"))

parser = argparse.ArgumentParser(
    description="Basic scene loading from NeuroforgeLab Interface"
)
parser.add_argument(
    "--asset_path",
    type=str,
    help="Path to assets",
)
parser.add_argument("--size", type=int, default=50, help="Size of the scene")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

simulation_app = AppLauncher(args_cli).app


from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.spot import SPOT_CFG

from forest_gen import ForestGenSpec


def run_simulation(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Function to run a simulation.
    Pretty basic as of now.

    Args:
        sim (sim_utils.SimulationContext): Simulation Context as defined by Isaac Lab
        scene (InteractiveScene): Scene generated from NFL Interface config class
    """
    sim_dt = sim.get_physics_dt()

    sim.pause()

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


if __name__ == "__main__":
    # Simulation Context initialization
    sim_cfg = SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    logger.debug("Simulation context initialized")

    # Camera setting
    sim.set_camera_view((0.0, 0.0, 5.0), (1.0, 1.0, 4.0))

    # Scene generation
    generator = ForestGenSpec(size=args_cli.size, asset_path=get_asset_path(args_cli))
    logger.debug("Scene generator initialized")
    scene_factory = generator.create_instance(
        num_envs=2, env_spacing=1.0
    )  # , debug_models=args_cli.debug_models)
    my_scene_cfg = scene_factory.get_scene(SPOT_CFG)

    logger.debug("Scene generated")
    # my_scene_cfg = InteractiveSceneCfg(1, 1)
    # my_scene_cfg = InteractiveSceneCfg(1, 1)
    my_scene = InteractiveScene(my_scene_cfg)

    # Reset simulation context - required by Isaac Lab
    sim.reset()

    # Run scene
    logger.info("Setup completed!")  # Yipeee! Cola trinken!
    run_simulation(sim, my_scene)
    simulation_app.close()
