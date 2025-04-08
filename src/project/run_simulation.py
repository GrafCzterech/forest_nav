import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Basic scene loading from NeuroforgeLab Interface")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn. Defaults to 1.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from forest_gen.scene import ForestGenSpec



def run_simulation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Function to run a simulation.

    Pretty basic as for now.
    Args:
        sim (sim_utils.SimulationContext): Simulation Context as defined by Isaac Lab
        scene (InteractiveScene): Scene generated from NFL Interface config class
    """
    sim_dt = sim.get_phisics_dt()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

def main():
    """Main sim function.
    """
    # Simulation Context initialization
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Camera setting
    sim.set_camera_view([0.0, 0.0, 5.0])

    # Scene generation
    generator = ForestGenSpec()
    scene_factory = generator.create_instance()
    my_scene_cfg = scene_factory.new_scene()  # TODO add the option to init multiple scenes - in NFL backend
    my_scene = InteractiveScene(my_scene_cfg)

    # Reset simulation context - required by Isaac Lab
    sim.reset()

    # Run scene
    print("[NFL INFO]: Setup completed!")
    run_simulation(sim, my_scene)

if __name__=="__main__":
    main()
    simulation_app.close()