import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing as tp
import utils
import math
from torch.distributions import Normal


class BackwardMap(nn.Module):
    def __init__(self, env_params, embed_dim, used_preference: bool = True):
        super(BackwardMap, self).__init__()
        self.layer_size = 512
        self.used_preference = used_preference
        if used_preference:
            self.fc1 = nn.Linear(env_params['obs'] + env_params['rewards'], self.layer_size)
        else:
            self.fc1 = nn.Linear(env_params['obs'], self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, self.layer_size)
        self.fc3 = nn.Linear(self.layer_size, self.layer_size)
        self.tanh = nn.Tanh()
        self.backward_out = nn.Linear(self.layer_size, embed_dim)
        self.embed_dim = embed_dim

        self.apply(utils.weight_init)

    def forward(self, obs, pref):
        if self.used_preference:
            x = torch.cat([obs, pref], dim=-1)
        else:
            x = obs
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        x = self.tanh(x)
        x = F.relu(self.fc3(x), inplace=True)
        B = self.backward_out(x)

        B = math.sqrt(self.embed_dim) * F.normalize(B, dim=1)
        return B


class ForwardMap(nn.Module):
    def __init__(self, env_params, embed_dim):
        super(ForwardMap, self).__init__()
        self.embed_dim = embed_dim
        self.layer_size = 512
        self.num_actions = env_params['action']
        self.fc1 = nn.Linear(env_params['obs'] + embed_dim, self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, self.layer_size)
        self.fc3 = nn.Linear(self.layer_size, self.layer_size)
        self.tanh = nn.Tanh()
        self.f1 = nn.Linear(self.layer_size, embed_dim * env_params['action'])
        self.f2 = nn.Linear(self.layer_size, embed_dim * env_params['action'])

        self.apply(utils.weight_init)

    def forward(self, obs, w):
        # w = w / torch.sqrt(1 + torch.norm(w, dim=-1,
        #                    keepdim=True) ** 2 / self.embed_dim)
        x = torch.cat([obs, w], dim=1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        x = self.tanh(x)
        x = F.relu(self.fc3(x), inplace=True)
        f1 = self.f1(x)
        f2 = self.f2(x)

        return f1.reshape(-1, self.embed_dim, self.num_actions), f2.reshape(-1, self.embed_dim, self.num_actions)
