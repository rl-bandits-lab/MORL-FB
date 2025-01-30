import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import random
import numpy as np
from module import ForwardMap, BackwardMap, Actor
from multi_step import MOMultiStepMemory
from utils import soft_update, reward2to3, reward2to4


class MORLAgent:
    def __init__(self, env, test_env, config, path, wandb=None) -> None:
        self.env = env
        self.test_env = test_env
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_max = self.env.action_space.high[0]
        self.action_min = self.env.action_space.low[0]
        self.reward_dim = self.env.reward_dim

        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.env.action_space.seed(0)

        self.wandb = wandb
        self.save = config['save']
        self.z_dim = config['z_dim']
        self.num_steps = config['num_steps']
        self.start_steps = config['start_steps']
        self.memory_size = config['memory_size']
        self.eval_steps = config['eval_steps']
        self.save_steps = config['save_steps']
        self.inference_size = config['interface_size']
        self.hidden_dim = config['hidden_dim']
        self.feature_dim = config['feature_dim']
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.gamma = config['gamma']
        self.lr = config['lr']
        self.delay_actor = config['delay_actor']
        self.update_per_step = config['update_per_step']
        self.clip = config['clip']
        self.expl_scale = config['expl_scale']
        self.policy_scale = config['policy_scale']
        self.q_loss_coef = config['q_loss_coef']
        self.seed = config['seed']

        self.tuning = config['tuning']
        self.her = config['her']
        self.constraint = config['constraint']

        if self.tuning:
            self.memory_reward = config['memory_reward_dim']
        else:
            self.memory_reward = self.reward_dim

        # record the evaluation results
        self.all_returns = []

        self.steps = 0
        self.episodes = 0
        self.learning_steps = 0
        self.eval_time = 0

        self.log_path = path
        if self.save and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.device = torch.device(
            f"cuda:{config['device_id']}" if torch.cuda.is_available() else "cpu")

        self.memory = MOMultiStepMemory(
            self.memory_size, self.env.observation_space.shape, self.memory_reward,
            self.env.action_space.shape, self.device, self.gamma)

        self.z_memory = MOMultiStepMemory(
            self.memory_size, self.env.observation_space.shape, self.memory_reward,
            self.env.action_space.shape, self.device, self.gamma)

        self.forward_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
                                      self.feature_dim, self.hidden_dim, preprocess=True, add_trunk=False).to(self.device)

        self.forward_target_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
                                             self.feature_dim, self.hidden_dim, preprocess=True, add_trunk=False).to(self.device)

        self.backward_net = BackwardMap(
            self.obs_dim, self.reward_dim, self.z_dim, self.hidden_dim, norm_z=True).to(self.device)

        self.backward_target_net = BackwardMap(
            self.obs_dim, self.reward_dim, self.z_dim, self.hidden_dim, norm_z=True).to(self.device)

        self.actor = Actor(self.obs_dim, self.z_dim, self.action_dim, self.feature_dim,
                           self.hidden_dim, preprocess=True, add_trunk=False, low=self.action_min, high=self.action_max).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.z_dim, self.action_dim, self.feature_dim,
                                  self.hidden_dim, preprocess=True, add_trunk=False, low=self.action_min, high=self.action_max).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr)
        self.f_optimizer = torch.optim.Adam(
            self.forward_net.parameters(), lr=self.lr)
        self.b_optimizer = torch.optim.Adam(
            self.backward_net.parameters(), lr=self.lr)
        
        # used when constraint is True
        self.pref_table = []

        for i in range(self.reward_dim):
            self.pref_table.append(np.eye(1, self.reward_dim, i)[0])

        self.pref_table.append(np.ones(self.reward_dim) / self.reward_dim)
        self.pref_table = np.array(self.pref_table)
        self.pref_table = self.pref_table.astype(np.float32)

        self.forward_net.train()
        self.backward_net.train()
        self.actor.train()
        self.actor_target.train()
        self.forward_target_net.train()
        self.backward_target_net.train()

        if self.reward_dim == 2:
            self.p_name = ['9010', '8020', '7030', '6040', '5050',
                           '4060', '3070', '2080', '1090']
            self.PREF = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5],
                         [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
        elif self.reward_dim == 3:
            self.p_name = ['Average', '811', '181', '118']
            self.PREF = [[0.333, 0.333, 0.334], [0.8, 0.1, 0.1], [
                0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        elif self.reward_dim == 5:
            self.p_name = ['Average', '61111',
                           '16111', '11611', '11161', '11116']
            self.PREF = [[0.2, 0.2, 0.2, 0.2, 0.2], [0.6, 0.1, 0.1, 0.1, 0.1], [
                0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.1, 0.6, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6]]

    def run(self) -> None:
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self) -> bool:
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps

    def get_action(self, state, z, eval=False) -> np.ndarray:
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        z = torch.Tensor(z).unsqueeze(0).to(self.device)
        stddev = self.expl_scale
        dist = self.actor(state, z, stddev)
        action = dist.sample()

        if eval:
            action = dist.mean

            action = action.cpu().numpy()[0]
        else:
            if self.start_steps > self.steps:
                action = self.env.action_space.sample()

            else:
                action = action.cpu().numpy()[0]

        action = action.clip(self.action_min, self.action_max)

        return action

    def get_target_action(self, state, z) -> torch.Tensor:
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        z = torch.Tensor(z).unsqueeze(0).to(self.device)

        stddev = self.policy_scale
        dist = self.actor_target(state, z, stddev)

        action = dist.sample(clip=self.clip)

        noise = torch.randn_like(action)
        noise = torch.clamp(noise, -0.25, 0.25)
        action = action + noise

        action = torch.clamp(action, self.action_min, self.action_max)

        return action

    def sample_z(self, size) -> torch.Tensor:
        gaussian_rdv = torch.randn(size, self.z_dim).to(self.device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
        z = math.sqrt(self.z_dim) * gaussian_rdv
        return z

    def get_pref(self) -> np.ndarray:

        if self.constraint:
            preference = self.pref_table[np.random.randint(
            0, self.pref_table.shape[0])]
        else:
            preference = np.random.dirichlet(np.ones(self.reward_dim))
            preference = preference.astype(np.float32)

        return preference
    
    @torch.no_grad()
    def inference_z(self, preference, eval=False) -> torch.Tensor:

        if self.steps < self.start_steps and not eval:
            z = self.sample_z(1)
            return z

        _, _, _, rewards, next_states, _, _ = self.z_memory.sample(
            self.inference_size)

        # using consine similarity to select close preference set
        if eval and not self.tuning and not self.her:
            _, _, _, rewards, next_states, _, _ = self.z_memory.sample(
                self.inference_size, eval=True, pref=preference)
            
        #tansform reward to 3 dimension or 4 dimension
        if self.tuning:
            if self.reward_dim == 3:
                rewards = reward2to3(next_states, rewards)
            elif self.reward_dim == 4:
                rewards = reward2to4(next_states, rewards)

        preference = torch.FloatTensor(preference).to(self.device)
        prefs = preference.unsqueeze(0).repeat(self.inference_size, 1)

        B = self.backward_net(next_states, prefs)

        dot_reward = torch.einsum('sd, sd -> s', rewards, prefs)
        dot_reward = dot_reward.unsqueeze(1)

        z = torch.matmul(dot_reward.T, B) / self.inference_size
        z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)

        return z

    def train_episode(self) -> None:
        self.episodes += 1
        current_steps = 0
        episode_reward = 0
        done = False
        state, _ = self.env.reset(seed=self.seed)

        preference = self.get_pref()
        z = self.inference_z(preference)
        z = z.squeeze().cpu().numpy()

        while (not done) and (current_steps < self.env._max_episode_steps):
            with torch.no_grad():
                action = self.get_action(state, z)
            next_state, reward, done, _, _ = self.env.step(action)

            self.steps += 1
            current_steps += 1
            episode_reward += reward

            masked_done = done

            self.memory.append(
                state, preference, action, reward, next_state, masked_done,
                episode_done=done, her=self.her)

            if not done:
                self.z_memory.append(
                    state, preference, action, reward, next_state, masked_done,
                    episode_done=done, her=self.her)

            state = next_state

            if self.is_update():
                if self.steps % self.update_per_step == 0:
                    self.learn()
                if self.steps % self.eval_steps == 0:
                    with torch.no_grad():
                        self.eval_time += 1
                        self.actor.eval()
                        for i in range(len(self.PREF)):
                            self.evaluation(i)
                        self.actor.train()

                if self.save and self.steps % self.save_steps == 0:
                    self.save_model(self.log_path, self.steps//self.save_steps)

    # used on calculating z for a batch of preference
    @torch.no_grad()
    def inference_z_batch(self, preference):

        _, _, _, rewards, next_states, _, _ = self.z_memory.sample(self.inference_size)
        
        rewards = rewards.unsqueeze(1).repeat(1, self.batch_size, 1)

        next_states_batch = next_states.repeat(self.batch_size, 1)

        with torch.no_grad():
            prefs_batch = preference.repeat(self.inference_size, 1)
            prefs = preference.unsqueeze(0).repeat(self.inference_size, 1, 1)

            B = self.backward_net(next_states_batch, prefs_batch)
            B = B.view(self.inference_size, self.batch_size, self.z_dim)

            dot_reward = torch.einsum('isd, isd -> is', rewards, prefs)

            z = torch.einsum('is, isk -> sk', dot_reward, B) / self.inference_size
            z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)

        return z

    def learn(self):
        states, preferences, actions, rewards, next_states, dones, _ = self.memory.sample(
            self.batch_size)
        
        if self.tuning:
            rewards = reward2to3(next_states, rewards)

        if self.her:
            zs = self.inference_z_batch(preferences)
        else:
            pref = self.get_pref()
            z = self.inference_z(pref)
            zs = z.repeat(self.batch_size, 1)
            zs = zs.to(self.device)
            preference = torch.FloatTensor(pref).to(self.device)
            preferences = preference.unsqueeze(0).repeat(self.batch_size, 1)

        self.update_fb(states, actions, rewards,
                       next_states, dones, zs, preferences)

        if self.learning_steps % self.delay_actor == 0:
            self.update_actor(states, zs)
            soft_update(self.forward_target_net, self.forward_net, self.tau)
            soft_update(self.backward_target_net, self.backward_net, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        self.learning_steps += 1

    def update_actor(self, states, zs):
        stddev = 0
        dist = self.actor(states, zs, stddev)
        actions = dist.mean

        F1, F2 = self.forward_net(states, zs, actions)
        Q = torch.einsum('sd, sd -> s', F1, zs)

        q_loss = -Q.mean()

        actor_loss = q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.wandb is not None:
            self.wandb.log(
                {"loss/Actor_loss": actor_loss.item(),
                 "loss/Actor_Q_loss": q_loss.item(),
                 "steps": self.steps, })

    def update_fb(self, states, actions, rewards, next_states, dones, zs, prefs):

        fb_loss = 0
        with torch.no_grad():
            next_actions = self.get_target_action(next_states, zs).squeeze()

            target_F1, target_F2 = self.forward_target_net(
                next_states, zs, next_actions)
            target_B = self.backward_target_net(next_states, prefs)

            target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)
            target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)

            target_M = torch.min(target_M1, target_M2)

        F1, F2 = self.forward_net(states, zs, actions)
        B = self.backward_net(next_states, prefs)
        M1 = torch.einsum('sd, td -> st', F1, B)
        M2 = torch.einsum('sd, td -> st', F2, B)
        I = torch.eye(*M1.size(), device=M1.device)

        off_diag = ~I.bool()

        dones = dones.view(-1, 1)

        fb_offdiag = 0.5 * sum((M - self.gamma * (1 - dones) * target_M)
                               [off_diag].pow(2).mean() for M in [M1, M2])

        fb_diag = -sum(((1-dones) * M).diag().mean() for M in [M1, M2])

        fb_loss = fb_offdiag + fb_diag

        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()

        orth_loss = orth_loss_diag + orth_loss_offdiag

        fb_loss += orth_loss

        with torch.no_grad():
            next_Q1, next_Q2 = [torch.einsum(
                'sd,sd->s', F, zs) for F in [target_F1, target_F2]]

            min_Q = torch.min(next_Q1, next_Q2)

            next_Q = (1.0 - dones).squeeze() * self.gamma * min_Q

            dot_reward = torch.einsum('sd, sd -> s', rewards, prefs)

            target_Q = dot_reward + next_Q

        Q1, Q2 = [torch.einsum('sd, sd->s', F, zs) for F in [F1, F2]]

        q_loss = F.mse_loss(Q1, target_Q, reduction='sum') + \
            F.mse_loss(Q2, target_Q, reduction='sum')

        q_loss = self.q_loss_coef * q_loss

        fb_loss += q_loss

        self.f_optimizer.zero_grad()
        self.b_optimizer.zero_grad()
        fb_loss.backward()
        self.f_optimizer.step()
        self.b_optimizer.step()

        if self.wandb is not None:
            self.wandb.log(
                {"loss/FB_loss": fb_loss.item(),
                 "loss/Orth_loss": orth_loss.item(),
                 "loss/FB_offdiag": fb_offdiag.item(),
                 "loss/FB_diag": fb_diag.item(),
                 "loss/Orth_diag": orth_loss_diag.item(),
                 "loss/Orth_offdiag": orth_loss_offdiag.item(),
                 "loss/FB Q_loss": q_loss.item(),
                 "steps": self.steps, })

    @torch.no_grad()
    def evaluation(self, preference_index):
        preference = self.PREF[preference_index]
        z = self.inference_z(preference, eval=True)
        z = z.squeeze().cpu().numpy()

        episode = 5
        episode_length = [0 for _ in range(episode)]
        returns = np.zeros((episode, self.reward_dim))
        for i in range(episode):
            current_steps = 0
            state, _ = self.test_env.reset(seed=i)
            done = False
            episode_reward = np.zeros(self.reward_dim)

            while (not done) and current_steps < self.env._max_episode_steps:

                current_steps += 1
                action = self.get_action(state, z, eval=True)
                next_state, reward, done, _, _ = self.test_env.step(action)

                episode_reward += reward
                state = next_state

            returns[i] = episode_reward
            episode_length[i] = current_steps

        dot_reward = np.dot(returns.mean(axis=0), preference)
        avg_length = np.mean(episode_length)

        self.all_returns.append(returns)

        if self.wandb is not None:
            self.wandb.log(
                {f"eval/{self.p_name[preference_index]}": dot_reward,
                 f"len/{self.p_name[preference_index]}_length": avg_length,
                 "steps": self.steps, })

        print('-' * 60)
        print(
            f'Eval Preference {preference} => update steps : {self.learning_steps}, returns : {returns.mean(axis=0)}, avg length : {avg_length}, dot reward : {dot_reward}')
        print('-' * 60)

    def save_model(self, path, eval_time):
        torch.save(self.actor.state_dict(), os.path.join(
            path, f'actor_{eval_time}.pth'))
        torch.save(self.forward_net.state_dict(), os.path.join(
            path, f'forward_net_{eval_time}.pth'))
        torch.save(self.backward_net.state_dict(), os.path.join(
            path, f'backward_net_{eval_time}.pth'))

        torch.save(self.forward_target_net.state_dict(), os.path.join(
            path, f'forward_target_net.pth'))
        torch.save(self.backward_target_net.state_dict(), os.path.join(
            path, f'backward_target_net.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(
            path, f'actor_target.pth'))

        torch.save(self.memory, os.path.join(path, f'memory.pth'))

        torch.save(self.z_memory, os.path.join(
            path, f'z_memory.pth'))
        torch.save(self.all_returns, os.path.join(
            path, f'all_returns.pth')
        )

    def load_model(self, path, num):
        self.actor.load_state_dict(torch.load(
            os.path.join(path, f'actor_{num}.pth'), map_location=f'cuda:{self.device}'))
        self.forward_net.load_state_dict(torch.load(
            os.path.join(path, f'forward_net_{num}.pth'), map_location=f'cuda:{self.device}'))
        self.backward_net.load_state_dict(torch.load(
            os.path.join(path, f'backward_net_{num}.pth'), map_location=f'cuda:{self.device}'))
        self.forward_target_net.load_state_dict(torch.load(
            os.path.join(path, 'forward_target_net.pth'), map_location=f'cuda:{self.device}'))
        self.backward_target_net.load_state_dict(torch.load(
            os.path.join(path, 'backward_target_net.pth'), map_location=f'cuda:{self.device}'))
        self.memory = torch.load(os.path.join(
            path, 'memory.pth'), map_location=f'cuda:{self.device}')
        self.z_memory = torch.load(os.path.join(
            path, 'z_memory.pth'), map_location=f'cuda:{self.device}')
        self.memory.device = self.device
        self.z_memory.device = self.device

    @torch.no_grad()
    def test(self, preference, episode=5):
        z = self.inference_z(preference, eval=True)
        z = z.squeeze().cpu().numpy()
        self.actor.eval()

        episode = episode
        returns = np.empty((episode, self.reward_dim))
        for i in range(episode):
            current_steps = 0
            state, _ = self.test_env.reset(seed=i)

            done = False
            episode_reward = np.zeros(self.reward_dim)

            while (not done) and current_steps < self.env._max_episode_steps:
                current_steps += 1
                action = self.get_action(state, z, eval=True)
                next_state, reward, done, _, _ = self.test_env.step(action)

                episode_reward += reward
                state = next_state

            self.test_env.close()
            returns[i] = episode_reward

        dot_reward = np.dot(returns.mean(axis=0), preference)

        return dot_reward, returns