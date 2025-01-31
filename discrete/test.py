from agent import DiscreteAgent
import mo_gymnasium
import os
import numpy as np
import time
import torch.nn.functional as F
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--env_name', type=str, default='deep-sea-treasure-v0')
parser.add_argument('--time_str', type=str, default='')
args = parser.parse_args()

if __name__ == '__main__':

    configs = {
        'num_steps': 1005000,
        'start_steps': 10000,
        'eval_steps': 10000,
        'save_steps': 50000,
        'memory_size': 1000000,
        'save': False,
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

    name = f'MORL-FB_{args.env_name}'


    path = os.path.join(f'log/{args.env_name}', f'{args.time_str}_{name}')
    agent = DiscreteAgent(env, test_env, configs, path)

    agent.load_model(path, 1000000)

    prefs = np.load(f'prefs/{args.env_name}.npy')


    all_rewards = []

    for p in tqdm(prefs):
        reward, _ = agent.test(p)
        all_rewards.append(reward)

    np.save(f'rewards/MORL-FB/{name}', all_rewards)

