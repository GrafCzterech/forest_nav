import os
import json

from manager import EnvCfg

import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

LOG_DIR = "logs"

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

    # TODO create an example config
    with open("agent.json", "r") as f:
        agent_cfg = json.load(f)

    agent_cfg = process_sb3_cfg(agent_cfg)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)

    # configure the logger

    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])

    agent.set_logger(new_logger)

    # callbacks for agent

    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=LOG_DIR, name_prefix="model", verbose=2
    )

    # train the agent

    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)

    # save the final model

    agent.save(os.path.join(LOG_DIR, "model"))

    # close the simulator

    env.close()
