from agent import DiscreteAgent
import mo_gymnasium
import os
import wandb
import numpy as np
import time
import torch.nn.functional as F
import torch

if __name__ == '__main__':
    # env_name = "mo-reacher-v4"
    # env_name = "deep-sea-treasure-v0"
    # env_name = "minecart-v0"
    # env_name = "mo-lunar-lander-v2"
    # env_name = "mo-highway-v0"
    env_name = "fruit-tree-v0"

    configs = {
        'num_steps': 1005000,
        'start_steps': 10000,
        'memory_size': 1000000,
        'save': True,
        'z_dim': 100,
        'interface_size': 1024,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.995,
        'lr': 3e-4,
        'seed': 10,
        'update_interval': 1,
    }

    env = mo_gymnasium.make(env_name, max_episode_steps=50)
    test_env = mo_gymnasium.make(env_name, max_episode_steps=50)

    time_str = time.strftime("%Y%m%d-%H%M%S")

    name = f'{env_name}_calculate_all_z_HER_no_goal_sum_boltzmann_epsilon_delay_update'

    project_name = 'FB_MORL_Discrete'

    wandb.init(project=project_name, name=f'{time_str}_{name}', config=configs)
    wandb.require("core")

    path = os.path.join(f'log/{env_name}', time_str)
    path = os.path.join(path, name)

    agent = DiscreteAgent(env, test_env, configs, path, wandb)

    agent.run()
