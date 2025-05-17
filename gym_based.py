from manager import EnvCfg

import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

# https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html

if __name__ == "__main__":
    env_cfg = EnvCfg()
    gymnasium.register(
        id="env",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
    )
    env = gymnasium.make("env", cfg=env_cfg)
    env.reset()
    env.step(env.action_space.sample())
    env.close()
    # https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html
