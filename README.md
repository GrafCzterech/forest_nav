# NFL-Lab-Bridge

## Yup, that's me, the bridge between Isaac Lab and NeuroforgeLab

Use ```install.sh``` to setup the virtual environment and start working with
this bridge. As said in forest-gen's ```README.md```:
> Lord have mercy upon those, who wish to run it on windows (at least for now).

NOTE: Isaac Lab and its required packages are not installed by aforementioned
script.

[Spot USD path](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/BostonDynamics/spot/spot.usd)

## Scripts

Since we all have dementia, the command to launch IsaacLab coupled with the
arguments to run different versions of engine are stored in scripts.

### `sim.sh`

This script launches Isaac Lab with the most bare-bones setup. It doesn't have
any of the robot observations, actions configured. It's just a simple scene
initialized from `SceneCfgFactory`. If you want to test if the scene is
well generated, this is the script to use.

### `manager.sh`

This script launches Isaac Lab via a `ManagerBasedRLEnvCfg`. It has the robot
configured and the scene is initialized through `SceneCfgFactory` and reset
via `SceneSpec.create_reset_event()`. If you want to test if spot
is well configured, and the scene can reset itself, this is the script
to use.

### `train.sh`

This script launches RL training utilizing the `ManagerBasedRLEnvCfg`.
It essentially builds on top of `manager.sh` and adds the training
part.
