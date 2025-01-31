import numpy as np
from environments.humanoid_v4 import HumanoidEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from gymnasium import register


class MOHumanoid5dEnv(HumanoidEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HumanoidEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/humanoid/) for more information.

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Reward for running forward (x-velocity)
    - 1: Control cost of the action
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(5,))
        self.reward_dim = 5

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        x_velocity = info["x_velocity"]
        y_velocity = info["y_velocity"]
        left_elbow = info["left_elbow_velocity"] * 0.15
        right_elbow = info["right_elbow_velocity"] * 0.15
        negative_cost = 10 * info["reward_quadctrl"]

        vec_reward = np.array([x_velocity, y_velocity, left_elbow, right_elbow, negative_cost], dtype=np.float32)

        vec_reward += self.healthy_reward  # All objectives are penalyzed when the agent falls

        return observation, vec_reward, terminated, truncated, info


register(id='mo-humanoid5d-v0', entry_point='environments.mo_humanoid5d:MOHumanoid5dEnv', max_episode_steps=1000,)