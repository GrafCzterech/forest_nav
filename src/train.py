#  DO NOT USE RN

import os
import logging

from manager import EnvCfg

import gymnasium as gym
from datetime import datetime
import random

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

# this is THE script. John Script.

# config shortcuts
LOG_DIR = "../logs"
algorithm = "ppo"
agent_cfg_entry_point = (
    "skrl_cfg_entry_point"
    if algorithm in ["ppo"]
    else f"skrl_{algorithm}_cfg_entry_point"
)
N_TIMESTEPS = 10**6  # number of timesteps to train for

# https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html

if __name__ == "__main__":
    # first setup the config
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = 8  # lets try this many for now
    # it best be registered in gymnasium
    gym.register(
        id="env",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
    )
    env = gym.make("env", cfg=env_cfg, render_mode="rgb_array")

    # video output
    video_kwargs = {
        "video_folder": os.path.join(LOG_DIR, "videos", "train"),
        "step_trigger": lambda step: step % 2000 == 0,
        "video_length": 200,  # 200 and 2000 above can be changed, these are default values from scripts
        "disable_logger": True,
    }
    logging.debug("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # then wrap it for some reason
    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # type: ignore

    logging.debug("environment created")

    # then initialize the policy optimizer
    agent = PPO(POLICY_ARCH, env, verbose=1)

    # configure the logger
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=LOG_DIR, name_prefix="model", verbose=2
    )

    # train the agent
    logging.debug("Training agent")
    agent.learn(total_timesteps=N_TIMESTEPS, callback=checkpoint_callback)

    # save the final model
    logging.debug("Saving agent")
    agent.save(os.path.join(LOG_DIR, "model"))

    # close the simulator
    env.close()
