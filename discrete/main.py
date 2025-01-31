from agent import DiscreteAgent
import mo_gymnasium
import os
import wandb
import numpy as np
import time
import torch.nn.functional as F
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--cuda_device', type=int, default=0, help='cuda device id')
parser.add_argument('--env_name', type=str, default='deep-sea-treasure-v0', help='environment name')
parser.add_argument('--project_name', type=str, default='MORL-FB', help='wandb project name')
args = parser.parse_args()


if __name__ == '__main__':

    configs = {
        'num_steps': 1005000,
        'start_steps': 10000,
        'eval_steps': 10000,
        'save_steps': 50000,
        'memory_size': 1000000,
        'save': True,
        'z_dim': 100,
        'interface_size': 1024,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.995,
        'lr': 3e-4,
        'seed': args.seed,
        'update_interval': 5,
        'her': True
    }

    env = mo_gymnasium.make(args.env_name, max_episode_steps=50)
    test_env = mo_gymnasium.make(args.env_name, max_episode_steps=50)

    time_str = time.strftime("%Y%m%d-%H%M%S")

    name = f'MORL-FB_{args.env_name}'


    wandb.init(project=args.project_name, name=f'{time_str}_{name}', config=configs)

    path = os.path.join(f'log/{args.env_name}', f'{time_str}_{name}')

    agent = DiscreteAgent(env, test_env, configs, path, wandb)

    agent.run()
