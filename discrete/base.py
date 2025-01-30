import numpy as np
import torch


class MOMemory:

    def __init__(self, capacity, state_shape, reward_shape, action_shape, device):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.reward_shape = reward_shape
        self.preference_shape = reward_shape
        self.device = device
        self.is_image = len(state_shape) == 3
        self.state_type = np.uint8 if self.is_image else np.float32

        self.reset()

    def append(self, state, preference, action, reward, next_state, done,
               episode_done=None):
        self._append(state, preference, action, reward, next_state, done)

    def _append(self, state, preference, action, reward, next_state, done):
        state = np.array(state, dtype=self.state_type)
        next_state = np.array(next_state, dtype=self.state_type)
        reward = np.array(reward, dtype=np.float32)

        self.states[self._p] = state
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.preferences[self._p] = preference
        self.next_states[self._p] = next_state
        self.dones[self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity


    def sample(self, batch_size, inference=False, eval=False, half=False, pref=np.array([0.5, 0.5])):
        tmp = []
        if eval:
            # calculate the cosine similarity between the preference and the preference of the samples
            for i in range(self._n):
                tmp.append(np.dot(self.preferences[i], pref) /
                           (np.linalg.norm(self.preferences[i]) * np.linalg.norm(pref)))
            indices = np.argsort(tmp)[-batch_size:]
        elif inference:

            # sample from done with true
            indices = np.random.choice(
                np.where(self.dones == 1)[0], batch_size, replace=True)
        elif half:
            # sample half batch size from done with true half from random
            indices = np.random.choice(
                np.where(self.dones == 1)[0], batch_size//2, replace=True)
            indices = np.concatenate(
                [indices, np.random.randint(low=0, high=self._n, size=batch_size//2)])
        else:
            indices = np.random.randint(low=0, high=self._n, size=batch_size)

        return self._sample(indices)

    def _sample(self, indices):
        states = self.states[indices]
        next_states = self.next_states[indices]
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        actions = torch.Tensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        preferences = torch.FloatTensor(
            self.preferences[indices]).to(self.device)

        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, preferences, actions, rewards, next_states, dones, dones

    def __len__(self):
        return self._n

    def reset(self):
        self._n = 0
        self._p = 0

        self.states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type)
        self.actions = np.empty(
            (self.capacity, *self.action_shape), dtype=np.int64)
        self.rewards = np.empty(
            (self.capacity, self.reward_shape), dtype=np.float32)
        self.preferences = np.empty(
            (self.capacity, self.reward_shape), dtype=np.float32)
        self.next_states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type)
        self.dones = np.empty(
            (self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid], self.preferences[valid], self.actions[valid], self.rewards[valid],
            self.next_states[valid], self.dones[valid])

    def load(self, batch):
        num_data = len(batch[0])

        if self._p + num_data <= self.capacity:
            self._insert(
                slice(self._p, self._p+num_data), batch,
                slice(0, num_data))
        else:
            mid_index = self.capacity-self._p
            end_index = num_data - mid_index
            self._insert(
                slice(self._p, self.capacity), batch,
                slice(0, mid_index))
            self._insert(
                slice(0, end_index), batch,
                slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, mem_indices, batch, batch_indices):
        states, preferences, actions, rewards, next_states, dones = batch
        self.states[mem_indices] = states[batch_indices]
        self.preferences[mem_indices] = preferences[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]

