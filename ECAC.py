import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from utils import soft_update
from replay_memory import ReplayMemory
from model import MujocoDoubleQNetwork, MujocoPolicy
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

class ECACAgent(object):
    def __init__(self, env, test_env, log_dir, batch_size=64, lr=0.0003, 
                 memory_size=1000000, hidden_dim=256,
                 gamma=0.99, tau=0.005, limit_kl=False, kl_target=0.002, 
                 update_interval=4, Q_updates_per_step=2, num_steps=1000001, 
                 max_episode_steps=1000, log_interval=10, cuda=True):

        self.env = env
        self.test_env = test_env
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.Q_updates_per_step = Q_updates_per_step
        self.num_steps = num_steps
        self.log_interval = log_interval
        self.device = torch.device("cuda" if cuda else "cpu") 
        self.max_episode_steps = max_episode_steps
        self.max_grad_norm = 0.5
        self.target_kl= kl_target
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()/2.0
        print(f"target_kl: {self.target_kl}")
        self.limit_kl = limit_kl
        if self.limit_kl:
            self.log_alpha = torch.ones(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.log_rho = torch.ones(1, requires_grad=True, device=self.device)
            self.rho_optim = Adam([self.log_rho], lr=lr)
        else:
            self.alpha = 0
            self.rho = 0
        # Specify the directory for logging.
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        # Create tensorboard.
        self.writer = SummaryWriter(log_dir=self.summary_dir)

        # Create replay buffer.
        self.memory = ReplayMemory(memory_size, self.device) 

        # Create critic networks.
        self.critic_online = MujocoDoubleQNetwork(env.observation_space.shape[0], 
                env.action_space.shape[0], hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic_online.parameters(), lr=lr)
        self.critic_target = MujocoDoubleQNetwork(env.observation_space.shape[0], 
                env.action_space.shape[0], hidden_dim).to(device=self.device).eval()
        self.update_target()

        # Record the scale and bias of action space.
        self.action_scale = torch.FloatTensor(
                (env.action_space.high - env.action_space.low) / 2.).to(self.device)
        self.action_bias = torch.FloatTensor(
                (env.action_space.high + env.action_space.low) / 2.).to(self.device)

        # Create policy networks.
        self.policy = MujocoPolicy(env.observation_space.shape[0], env.action_space.shape[0],
                 self.action_scale, self.action_bias, hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.old_policy = MujocoPolicy(env.observation_space.shape[0], env.action_space.shape[0],
                 self.action_scale, self.action_bias, hidden_dim).to(self.device).eval()
        self.backup_policy()
        
        self.learning_steps = 0
        self.best_eval_score = -np.inf

    def backup_policy(self):
        self.old_policy.load_state_dict(self.policy.state_dict())

    def update_target(self):
        self.critic_target.load_state_dict(self.critic_online.state_dict())

    def explore(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.sample_random_action(state)
        return action.detach().cpu().numpy()[0]

    def exploit(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.greedy_action(state)
        return action.detach().cpu().numpy()[0]

    def calc_current_q(self, states, actions):
        curr_q1, curr_q2 = self.critic_online(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, rewards, next_states, masks):
        with torch.no_grad():
            next_action_mean= self.policy.sample_random_action(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_action_mean)
            next_q = torch.min(next_q1, next_q2)
        return rewards + masks * self.gamma * next_q

    def calc_critic_loss(self, states, actions, rewards, next_states, masks):
        curr_q1, curr_q2 = self.calc_current_q(states, actions)
        target_q = self.calc_target_q(rewards, next_states, masks)

        # Critic loss is mean squared TD errors.
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        return q1_loss, q2_loss

    def calc_policy_loss(self, states, updates):
        actions, entropies, cross_entropies = self.policy.sample_random_action(states, self.old_policy) # Sample actions for training policy.
        qf1, qf2 = self.critic_online(states, actions)
        policy_grad_term = (-qf1).mean()
        kl = cross_entropies - entropies
        if self.limit_kl:
            #self.alpha = 0
            self.alpha = torch.exp(self.log_alpha).detach()
            self.rho = torch.exp(self.log_rho).detach()
            policy_loss = policy_grad_term + self.alpha * cross_entropies - self.rho * entropies
        else:
            policy_loss = policy_grad_term
        return policy_loss, kl.detach(), entropies.detach(), cross_entropies.detach(), policy_grad_term.detach()

    def update_parameters(self):
        self.learning_steps += 1
        # Update Q networks.
        for i in range(self.Q_updates_per_step):
            # Sample a batch from memory.
            states, actions, rewards, next_states, masks = self.memory.sample(
                batch_size=self.batch_size)
            self.critic_optim.zero_grad()
            qf1_loss, qf2_loss = self.calc_critic_loss(states, actions, rewards, 
                next_states, masks)
            qf1_loss.backward()
            qf2_loss.backward()
            clip_grad_norm_(self.critic_online.parameters(), max_norm=self.max_grad_norm)
            self.critic_optim.step()

        policy_loss, kl, entropies, cross_entropies, policy_term = self.calc_policy_loss(states, self.learning_steps)    
        # Update coefficents
        if self.limit_kl:
            self.alpha_optim.zero_grad()
            alpha_loss = self.log_alpha*((self.target_kl-self.target_entropy)-cross_entropies)
            alpha_loss.backward()
            self.alpha_optim.step()     
            self.alpha_optim.zero_grad()
            rho_loss = self.log_rho*(entropies-self.target_entropy)
            rho_loss.backward()
            self.rho_optim.step()  
            self.rho_optim.zero_grad()
        # Update policy networks.
        self.backup_policy()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
        self.policy_optim.step()

        # Soft update target critic network.
        soft_update(self.critic_target, self.critic_online, self.tau)
        # Log training information.
        if self.learning_steps % self.log_interval == 0 and self.learning_steps > 100:
            self.writer.add_scalar(
                'loss/Q1', qf1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/kl', kl, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropies', entropies, self.learning_steps)
            self.writer.add_scalar(
                'stats/cross_entropies', cross_entropies, self.learning_steps)

    def evaluate(self, num_episodes=1):
        num_steps = 0
        total_return = 0.0
        for i_episode in range(num_episodes):
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                action = self.exploit(state)
                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state
            total_return += episode_return
        mean_return = total_return/num_episodes
        return mean_return

    def save_models(self, save_dir):
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.critic_online.save(os.path.join(save_dir, 'critic_online.pth'))
        self.critic_target.save(os.path.join(save_dir, 'critic_target.pth'))

