import mo_gymnasium
import os
from mo_agent import MORLAgent
import numpy as np
import time
import tqdm
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--env_name', type=str, default='mo-halfcheetah-v4')
parser.add_argument('--time_str', type=str, default='')
args = parser.parse_args()


if __name__ == '__main__':

    configs = {
        'num_steps': 3000000,
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

    name = f'MORL-FB_{args.env_name}'


    path = os.path.join(f'log/{args.env_name}', f'{args.time_str}_{name}')
    agent = MORLAgent(env, test_env, configs, path=path, wandb=None)

    agent.load_model(path, 3000000)

    prefs = np.load(f'prefs/{args.env_name}.npy')

    all_rewards = []

    for p in tqdm(prefs):
        reward = agent.test(p)
        all_rewards.append(reward)

    np.save(f'rewards/MORL-FB/{name}.npy', all_rewards)

