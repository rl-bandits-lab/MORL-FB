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
parser.add_argument('--cuda_device', type=int, default=0, help='cuda device id')
parser.add_argument('--env_name', type=str, default='deep-sea-treasure-v0', help='environment name')
parser.add_argument('--model_name', type=str, default='', help='model directory name')
parser.add_argument('--steps', type=int, default=1000000, help='model steps for file name')
parser.add_argument('--output_path', type=str, default='rewards/MORL-FB/output.npy', help='path for saving testing results')
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


    path = os.path.join(f'log/{args.env_name}', f'{args.model_name}')
    agent = DiscreteAgent(env, test_env, configs, path)

    agent.load_model(path, args.steps)

    prefs = np.load(f'prefs/{args.env_name}.npy')

    all_rewards = []

    for p in tqdm(prefs):
        reward, _ = agent.test(p)
        all_rewards.append(reward)

    np.save(f'{args.output_path}', all_rewards)

