from manager import EnvCfg

import gymnasium

# https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html

if __name__ == "__main__":
    env_cfg = EnvCfg()
    # env_cfg.sim.device = "cpu"
    gymnasium.register(
        id="env",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
    )
    env = gymnasium.make("env", cfg=env_cfg)
    env.reset()  # must be called before step
    env.step(env.action_space.sample())
    env.close()
