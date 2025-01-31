import mo_gymnasium
import os
import wandb
from mo_agent import MORLAgent
import numpy as np
import time
import torch.nn.functional as F
import torch
import argparse
from environments import mo_hopper2d, mo_hopper4d, mo_humanoid5d

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--cuda_device', type=int, default=0, help='cuda device id')
parser.add_argument('--env_name', type=str, default='mo-halfcheetah-v4', help='environment name')
parser.add_argument('--project_name', type=str, default='MORL-FB', help='wandb project name')
args = parser.parse_args()

if __name__ == '__main__':

    configs = {
        'num_steps': 3010000,
        'start_steps': 10000,
        'memory_size': 1000000,
        'save': True,
        'save_steps': 50000,
        'eval_steps': 10000,
        'z_dim': 150,
        'interface_size': 1024,
        'hidden_dim': 1024,
        'feature_dim': 512,
        'batch_size': 256,
        'tau': 0.01,
        'gamma': 0.99,
        'lr': 1e-4,
        'delay_actor': 5,
        'update_per_step': 2,
        'expl_scale': 0.1,
        'policy_scale': 0.2,  # schedlue
        'clip': 0.5,  # clip on truncated normal distsibution noise
        'q_loss_coef': 1,
        'tuning': False,
        'memory_reward_dim': 3, # Replay buffer reward dimension, used when tuning is True
        'her': False,
        'constraint': False,
        'device_id': args.cuda_device,
        'seed': args.seed,
    }
    
    if args.env_name == 'mo-halfcheetah-v4':
        env = mo_gymnasium.make(args.env_name, max_episode_steps=1000)
        test_env = mo_gymnasium.make(args.env_name, max_episode_steps=1000)
    else:
        env = mo_gymnasium.make(args.env_name, healthy_reward=1.0, max_episode_steps=1000)
        test_env = mo_gymnasium.make(args.env_name, healthy_reward=1.0, max_episode_steps=1000)

    time_str = time.strftime("%Y%m%d-%H%M%S")

    name = f'MORL-FB_{args.env_name}'

    wandb.init(project=args.project_name, name=f'{time_str}_{name}', config=configs)

    path = os.path.join(f'log/{args.env_name}', f'{time_str}_{name}')

    agent = MORLAgent(env, test_env, configs, path=path, wandb=wandb)

    agent.run()
