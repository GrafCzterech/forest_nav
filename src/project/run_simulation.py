import argparse
import logging

# TODO we need a centralized logging system
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] {%(module)s:%(lineno)d} %(levelname)s: %(message)s",
)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Basic scene loading from NeuroforgeLab Interface"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to spawn. Defaults to 1.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, SimulationCfg
import isaacsim.core.utils.prims as prims_utils
from pxr import PhysxSchema
from forest_gen.scene import ForestGenSpec

BUFFER_SIZE = 400000

def run_simulation(sim: SimulationContext, scene: InteractiveScene):
    """Function to run a simulation.

    Pretty basic as for now.
    Args:
        sim (sim_utils.SimulationContext): Simulation Context as defined by Isaac Lab
        scene (InteractiveScene): Scene generated from NFL Interface config class
    """
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

def resize_buffer(size: int, physics_scene_prim_path: str):
    """Function to resize """
    physics_scene_prim = prims_utils.create_prim(
        prim_path=physics_scene_prim_path
    )

    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    physxSceneAPI.CreateGpuTempBufferCapacityAttr(size)        
    physxSceneAPI.CreateGpuHeapCapacityAttr(size*4)


def main():
    """Main sim function."""
    # Simulation Context initialization
    sim_cfg = SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    logging.debug("Simulation context initialized")

    # resize_buffer(BUFFER_SIZE, "/World/physicsScene")  # TODO - Do something about it, doesn't work here right now. Try in scene generator

    # Camera setting
    sim.set_camera_view((0.0, 0.0, 5.0), (1.0, 1.0, 4.0))

    # Scene generation
    generator = ForestGenSpec(size=50)
    logging.debug("Scene generator initialized")
    scene_factory = generator.create_instance()
    my_scene_cfg = (
        scene_factory.new_scene()
    )  # TODO add the option to init multiple scenes - in NFL backend
    logging.debug("Scene generated")
    my_scene = InteractiveScene(my_scene_cfg)

    # Reset simulation context - required by Isaac Lab
    sim.reset()

    # Run scene
    logging.info("Setup completed!")  # Yipeee! Cola trinken!
    run_simulation(sim, my_scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
