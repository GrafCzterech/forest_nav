import os
import logging

from manager import EnvCfg

import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

LOG_DIR = "logs"
POLICY_ARCH = "MlpPolicy"  # or "CnnPolicy" for image-based policies
N_TIMESTEPS = 10**6  # number of timesteps to train for

# https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html

if __name__ == "__main__":
    env_cfg = EnvCfg()
    gymnasium.register(
        id="env",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
    )
    env = gymnasium.make("env", cfg=env_cfg)

    env = Sb3VecEnvWrapper(env)  # type: ignore

    logging.debug("environment created")

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
