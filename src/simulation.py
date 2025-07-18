from bridge_config import setup_logging, setup

logger = setup_logging(("forest_gen", "neuroforgelab", "main"))

simulation_app, args_cli = setup()

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, SimulationCfg

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
    generator = ForestGenSpec(size=100)
    logger.debug("Scene generator initialized")
    scene_factory = generator.create_instance()
    my_scene_cfg = scene_factory.new_scene(SPOT_CFG)

    logger.debug("Scene generated")
    my_scene = InteractiveScene(my_scene_cfg)

    # Reset simulation context - required by Isaac Lab
    sim.reset()

    # Run scene
    logger.info("Setup completed!")  # Yipeee! Cola trinken!
    run_simulation(sim, my_scene)
    simulation_app.close()
