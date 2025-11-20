import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils
import numpy as np
import typing as tp


class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = math.sqrt(self.dim) * F.normalize(x, dim=1)
        return y


def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")


def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)


class ForwardMap(nn.Module):
    """ forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_action_net = mlp(
                self.obs_dim + self.action_dim, hidden_dim, "ntanh", feature_dim, "layernorm", "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim,
                                 hidden_dim, "ntanh", feature_dim, "layernorm", "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim + self.action_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

        self.apply(utils.weight_init)

    def forward(self, obs, z, action):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)

        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1, F2


class BackwardMap(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, reward_dim, z_dim, hidden_dim, norm_z: bool = True, used_preference: bool = False) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        self.reward_dim = reward_dim

        self.used_preference = used_preference

        if self.used_preference:
            self.B = mlp(self.obs_dim+self.reward_dim, hidden_dim, "ntanh",
                        hidden_dim, "irelu", self.z_dim)
        else:
            self.B = mlp(self.obs_dim, hidden_dim, "ntanh",
                        hidden_dim, "irelu", self.z_dim)


        self.apply(utils.weight_init)

    def forward(self, obs, pref):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True
        if self.used_preference:
            x = torch.cat([obs, pref], dim=-1)
        else:
            x = obs
        B = self.B(x)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B


class Actor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True, low=-1.0, high=1.0) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess
        self.action_low = low
        self.action_high = high

        if self.preprocess:
            self.obs_net = mlp(self.obs_dim, hidden_dim,
                               "ntanh", feature_dim, "layernorm", "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim,
                                 hidden_dim, "ntanh", feature_dim, "layernorm", "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        self.policy = mlp(feature_dim, hidden_dim, "irelu", self.action_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, z, std):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            obs = self.obs_net(obs)
            h = torch.cat([obs, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        mu = self.policy(h)
        mu = torch.tanh(mu) * self.action_high
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(
            mu, std, low=self.action_low, high=self.action_high)
        return dist
