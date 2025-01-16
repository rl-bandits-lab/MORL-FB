from collections import deque
import numpy as np
from base import MOMemory


class MultiStepBuff:
    keys = ["state", "action", "reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.memory = {
            key: deque(maxlen=self.maxlen)
            for key in self.keys
        }

    def append(self, state, preference, action, reward):
        self.memory["state"].append(state)
        self.memory["preference"].append(preference)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        preference = self.memory["preference"].popleft()
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        return state, preference, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([
            r * (gamma ** i) for i, r
            in enumerate(self.memory["reward"])])

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in MultiStepBuff.')
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory['state'])


class MOMultiStepMemory(MOMemory):

    def __init__(self, capacity, state_shape, reward_shape, action_shape, device,
                 gamma=0.99):
        super(MOMultiStepMemory, self).__init__(
            capacity, state_shape, reward_shape, action_shape, device)
        self.gamma = gamma

    def append(self, state, preference, action, reward, next_state, done,
               episode_done=False):
        self._append(state, preference, action, reward, next_state, done)
        for _ in range(3):
            new_pref = self.get_pref()
            self._append(state, new_pref, action, reward, next_state, done)

    def get_pref(self) -> np.ndarray:
        preference = np.random.dirichlet(np.ones(self.reward_shape))
        preference = preference.astype(np.float32)

        return preference


class TrajectoryMemory():
    keys = ["state", "action", "reward", "next_state", "done"]

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {
            key: deque(maxlen=self.capacity)
            for key in self.keys
        }

    def append(self, state, action, reward, next_state, done):
        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)
        self.memory["next_state"].append(next_state)
        self.memory["done"].append(done)

    def get(self, num=0):
        assert len(self) >= num
        return self.memory['state'][num], self.memory['action'][num], \
            self.memory['reward'][num], self.memory['next_state'][num], \
            self.memory['done'][num]

    def __len__(self):
        return len(self.memory['state'])

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in TrajectoryMemory.')
        return self.memory[key]
