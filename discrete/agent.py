import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np
import random
import os
import itertools
import math
from module import ForwardMap, BackwardMap
from multi_step import MOMultiStepMemory
from utils import soft_update


class DiscreteAgent:
    def __init__(self, env, test_env, config, path, wandb=None):
        self.env = env
        self.test_env = test_env
        self.config = config
        self.wandb = wandb

        self.log_path = path

        self.steps = 0
        self.episodes = 0
        self.start_steps = config['start_steps']
        self.num_steps = config['num_steps']
        self.eval_steps = config['eval_steps']
        self.save_steps = config['save_steps']
        self.save = config['save']
        self.lr = config['lr']

        self.embed_dim = config['z_dim']
        self.env_params = {}
        self.env_params['obs'] = env.observation_space.shape[0]
        self.env_params['action'] = env.action_space.n
        self.env_params['rewards'] = env.reward_dim
        self.num_actions = self.env_params['action']
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.seed = config['seed']
        self.inference_size = config['interface_size']
        self.update_interval = config['update_interval']
        self.her = config['her']
        self.update_counter = 0


        if self.save and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)       

        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.env.action_space.seed(0)

        self.forward_map = ForwardMap(
            self.env_params, self.embed_dim).to(self.device)
        self.backward_map = BackwardMap(
            self.env_params, self.embed_dim).to(self.device)

        self.forward_target = ForwardMap(
            self.env_params, self.embed_dim).to(self.device)
        self.backward_target = BackwardMap(
            self.env_params, self.embed_dim).to(self.device)

        self.fb_optimizer = torch.optim.Adam(
            itertools.chain(self.forward_map.parameters(), self.backward_map.parameters()), lr=self.lr)

        self.memory = MOMultiStepMemory(config['memory_size'], self.env.observation_space.shape, self.env_params['rewards'],
                                        self.env.action_space.shape, self.device, self.gamma)
        
        if self.env_params['rewards'] == 2:
            self.p_name = ['9010', '8020', '7030', '6040', '5050',
                           '4060', '3070', '2080', '1090']
            self.PREF = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5],
                         [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
        elif self.env_params['rewards'] == 3:
            self.p_name = ['Average', '811', '181', '118']
            self.PREF = [[0.333, 0.333, 0.334], [0.8, 0.1, 0.1], [
                0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        elif self.env_params['rewards'] == 4:
            self.p_name = ['Average', '7111', '1711', '1171', '1117']
            self.PREF = [[0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1], [
                0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]
        elif self.env_params['rewards'] == 6:
            self.p_name = ['Average', '511111', '151111', '115111', '111511', '111151', '111115']
            self.PREF = [[0.166, 0.166, 0.166, 0.166, 0.166, 0.17], [0.5, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.5, 0.1, 0.1, 0.1, 0.1], [
                0.1, 0.1, 0.5, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.5, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.5, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]]     

    def run(self) -> None:
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def get_pref(self) -> np.ndarray:
        preference = np.random.dirichlet(np.ones(self.env_params['rewards']))
        preference = preference.astype(np.float32)

        return preference

    def train_episode(self) -> None:
        self.episodes += 1
        current_steps = 0
        episode_reward = 0
        done = False
        state, _ = self.env.reset(seed=self.seed)

        preference = self.get_pref()
        z = self.preference_guided_exploration(preference)

        while (not done) and current_steps < self.env._max_episode_steps:
            if self.steps < self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.get_action(state, z)
            next_state, reward, done, _, _ = self.env.step(action)

            self.steps += 1
            current_steps += 1
            self.update_counter += 1
            episode_reward += reward


            masked_done = done

            self.memory.append(
                state, preference, action, reward, next_state, masked_done,
                episode_done=done, her=self.her)

            state = next_state

            if self.is_update():
                if self.update_counter % self.update_interval == 0:
                    self.learn()
                if self.steps % self.eval_steps == 0:
                    for i in range(len(self.PREF)):
                        self.evaluation(i)
                    
                if self.save and self.steps % self.save_steps == 0:
                    self.save_model(self.log_path, self.steps)


    def is_update(self) -> bool:
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps

    def sample_z(self, size) -> torch.Tensor:
        gaussian_rdv = torch.randn(size, self.embed_dim).to(self.device)
        gaussian_rdv = Fun.normalize(gaussian_rdv, dim=1)
        z = math.sqrt(self.embed_dim) * gaussian_rdv
        return z

    @torch.no_grad()
    def preference_guided_exploration(self, preference, eval=False) -> torch.Tensor:

        if self.steps < self.start_steps and not eval:
            z = self.sample_z(1)
            return z

        _, _, _, rewards, next_states, _, _ = self.memory.sample(self.inference_size)

        preference = torch.FloatTensor(preference).to(self.device)
        prefs = preference.unsqueeze(0).repeat(self.inference_size, 1)

        B = self.backward_map(next_states, prefs)

        dot_reward = torch.einsum('sd, sd -> s', rewards, prefs)
        dot_reward = dot_reward.unsqueeze(1)

        z = torch.matmul(dot_reward.T, B) / self.inference_size
        z = math.sqrt(self.embed_dim) * Fun.normalize(z, dim=1)

        return z

    @torch.no_grad()
    def get_action(self, state, z, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        z = z.to(self.device)

        F1, F2 = self.forward_map(state, z)
        Q1, Q2 = [torch.einsum('sda,sd->sa', F, z) for F in [F1, F2]]
        Q = torch.min(Q1, Q2)

        if not eval and np.random.rand() < 0.35:
            action = self.env.action_space.sample()
        else:
            action = Q.max(1)[1].item()

        return action
    
    @torch.no_grad()
    def preference_guided_exploration_batch(self, preference) -> torch.Tensor:

        _, _, _, rewards, next_states, _, _ = self.memory.sample(
            self.inference_size)
        
        rewards = rewards.unsqueeze(1).repeat(1, self.batch_size, 1)

        next_states_batch = next_states.repeat(self.batch_size, 1)
        

        prefs_batch = preference.repeat(self.inference_size, 1)
        prefs = preference.unsqueeze(0).repeat(self.inference_size, 1, 1)

        B = self.backward_map(next_states_batch, prefs_batch)

        B = B.view(self.inference_size, self.batch_size, self.embed_dim)

        dot_reward = torch.einsum('isd, isd -> is', rewards, prefs)

        z = torch.einsum('is, isk -> sk', dot_reward, B) / self.inference_size
        z = math.sqrt(self.embed_dim) * Fun.normalize(z, dim=1)

        return z

    def learn(self) -> None:
        states, preferences, actions, rewards, next_states, dones, episode_dones = self.memory.sample(
            self.batch_size)
        
        if self.her:
            zs = self.preference_guided_exploration_batch(preferences)
        else:
            pref = self.get_pref()
            z = self.preference_guided_exploration(pref)
            zs = z.repeat(self.batch_size, 1)
            zs = zs.to(self.device)
            preference = torch.FloatTensor(pref).to(self.device)
            preferences = preference.unsqueeze(0).repeat(self.batch_size, 1)

        self.update_fb(states, actions, rewards,
                       next_states, dones, episode_dones, zs, preferences)

        soft_update(self.forward_target, self.forward_map, self.tau)
        soft_update(self.backward_target, self.backward_map, self.tau)

    def update_fb(self, states, actions, rewards, next_states, dones, episode_dones, zs, prefs):
        fb_loss = 0

        with torch.no_grad():
            target_F1, target_F2 = self.forward_target(next_states, zs)
            target_B = self.backward_target(next_states, prefs)
            next_Q1, next_Q2 = [torch.einsum(
                'sda,sd->sa', F, zs) for F in [target_F1, target_F2]]
            next_Q = torch.min(next_Q1, next_Q2)

            pi = Fun.softmax(next_Q/0.1, dim=1)

            target_F1, target_F2 = [torch.einsum('sa, sda -> sd', pi, F) for F in [target_F1, target_F2]]

            next_Q = torch.einsum('sa, sa -> s', pi, next_Q)

            target_M1, target_M2 = [torch.einsum(
                'sd, td->st', F, target_B) for F in [target_F1, target_F2]]

            target_M = torch.min(target_M1, target_M2)

            dot_reward = torch.einsum('sd, sd -> s', rewards, prefs)

            target_Q = dot_reward + \
                (1.0 - dones).squeeze() * self.gamma * next_Q

        F1, F2 = self.forward_map(states, zs)
        B = self.backward_map(next_states, prefs)

        idxs = actions[:, None].repeat(1, self.embed_dim)[
            :, :, None].long()

        F1, F2 = [F.gather(-1, idxs).squeeze()
                  for F in [F1, F2]]

        M1 = torch.einsum('sd, td->st', F1, B)
        M2 = torch.einsum('sd, td->st', F2, B)

        Q1, Q2 = [torch.einsum('sd, sd->s', F, zs) for F in [F1, F2]]


        Q_loss = Fun.mse_loss(Q1, target_Q, reduction='sum') + Fun.mse_loss(Q2, target_Q, reduction='sum')

        I = torch.eye(*M1.size(), device=M1.device)
        off_diag = ~I.bool()
        fb_offdiag = 0.5 * \
            sum((M - self.gamma * target_M)
                [off_diag].pow(2).mean() for M in [M1, M2])
        fb_diag = -sum(M.diag().mean()
                       for M in [M1, M2])

        measure_loss = fb_offdiag + fb_diag


        Cov = torch.matmul(B, B.T)

        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()       

        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()

        orth_loss = orth_loss_diag + orth_loss_offdiag


        fb_loss += Q_loss

        fb_loss += measure_loss

        fb_loss += orth_loss

        self.fb_optimizer.zero_grad()
        fb_loss.backward()
        self.fb_optimizer.step()

        if self.wandb is not None:
            self.wandb.log(
                {"loss/Q": Q_loss.item(),
                 "loss/fb_offdiag": fb_offdiag.item(),
                 "loss/fb_diag": fb_diag.item(),
                 "loss/measure": measure_loss.item(),
                 "loss/fb": fb_loss.item(),
                 "loss/orth_diag": orth_loss_diag.item(),
                "loss/orth_offdiag": orth_loss_offdiag.item(),
                 "loss/orth": orth_loss.item(),
                 "steps": self.steps, })
            

    @torch.no_grad()
    def evaluation(self, preference_index):
        preference = self.PREF[preference_index]
        episode = 5
        returns = np.zeros((episode, self.env_params['rewards']))

        for i in range(episode):
            z = self.preference_guided_exploration(preference, eval=True)
            state, _ = self.test_env.reset(seed=i)
            done = False
            episode_reward = np.zeros(self.env_params['rewards'])
            current_steps = 0

            while (not done) and current_steps < self.test_env._max_episode_steps:
                action = self.get_action(state, z, eval=True)
                next_state, reward, done, _, _ = self.test_env.step(action)
                state = next_state
                episode_reward += reward
                current_steps += 1

            returns[i] = episode_reward

        dot_reward = np.dot(returns.mean(axis=0), preference)

        if self.wandb is not None:
            self.wandb.log(
                {f"eval/{self.p_name[preference_index]}": dot_reward,
                 "steps": self.steps, })

        print(
            f'Preference: {preference} => Reward: {dot_reward}, Returns: {returns.mean(axis=0)}')

    def save_model(self, path, time):
        torch.save(self.forward_map.state_dict(),
                   os.path.join(path, f'forward_{time}.pth'))
        torch.save(self.backward_map.state_dict(),
                   os.path.join(path, f'backward_{time}.pth'))
        torch.save(self.forward_target.state_dict(),
                     os.path.join(path, f'forward_target_{time}.pth'))
        torch.save(self.backward_target.state_dict(),
                        os.path.join(path, f'backward_target_{time}.pth'))
        torch.save(self.memory, os.path.join(path, f'memory.pth'))

    def load_model(self, path, time):
        self.forward_map.load_state_dict(
            torch.load(os.path.join(path, f'forward_{time}.pth')))
        self.backward_map.load_state_dict(
            torch.load(os.path.join(path, f'backward_{time}.pth')))
        self.forward_target.load_state_dict(
            torch.load(os.path.join(path, f'forward_target_{time}.pth')))
        self.backward_target.load_state_dict(
            torch.load(os.path.join(path, f'backward_target_{time}.pth')))
        self.memory = torch.load(os.path.join(path, f'memory.pth'))

    def test(self, preference):
        episode = 5
        returns = np.zeros((episode, self.env_params['rewards']))

        for i in range(episode):
            z = self.preference_guided_exploration(preference, eval=True)
            state, _ = self.test_env.reset(seed=i)
            done = False
            episode_reward = np.zeros(self.env_params['rewards'])
            current_steps = 0

            while (not done) and current_steps < self.test_env._max_episode_steps:
                action = self.get_action(state, z, eval=True)
                next_state, reward, done, _, _ = self.test_env.step(action)
                state = next_state
                episode_reward += reward
                current_steps += 1

            returns[i] = episode_reward

        dot_reward = np.dot(returns.mean(axis=0), preference)

        print(
            f'Preference: {preference} => Reward: {dot_reward}, Returns: {returns.mean(axis=0)}')
        
        return returns, dot_reward