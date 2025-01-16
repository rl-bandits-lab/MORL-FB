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
    env_name = "deep-sea-treasure-v0"
    # env_name = "minecart-v0"
    # env_name = "mo-lunar-lander-v2"
    # env_name = "mo-highway-v0"

    configs = {
        'num_steps': 1505000,
        'start_steps': 10000,
        'memory_size': 1000000,
        'save': False,
        'z_dim': 150,
        'interface_size': 1024,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'lr': 3e-4,
        'seed': 10,
        'update_interval': 3,
    }

    env = mo_gymnasium.make(env_name, max_episode_steps=50)
    test_env = mo_gymnasium.make(env_name, max_episode_steps=50)

    time_str = "20250112-220823"

    name = f'{env_name}_calculate_all_z_HER_no_goal_sum_boltzmann_epsilon_delay_update'

    project_name = 'FB_MORL_Discrete'


    path = os.path.join(f'log/{env_name}', time_str)
    path = os.path.join(path, name)

    agent = DiscreteAgent(env, test_env, configs, path)

    agent.load_model(path, 130)

    p_name = ['9010', '8020', '7030', '6040', '5050',
                           '4060', '3070', '2080', '1090']
    PREF = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5],
                         [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]

    for i in range(9):
        agent.test(PREF[i])


