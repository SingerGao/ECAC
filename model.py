import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import weights_init_


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class MLPBase(BaseNetwork):
    def __init__(self, num_inputs, num_ouputs, hidden_dim=256):
        super(MLPBase, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_ouputs)

    def forward(self, x):
        x = F.relu(self.linear1(x), inplace=False)
        x = F.relu(self.linear2(x), inplace=False)
        return x

class MujocoDoubleQNetwork(BaseNetwork):
    def __init__(self, num_obs, num_actions, hidden_dim=256):
        super(MujocoDoubleQNetwork, self).__init__()
        # Q1 architecture
        self.q1_base = MLPBase(num_obs+num_actions, hidden_dim)
        self.q1_out_layer = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.q2_base = MLPBase(num_obs+num_actions, hidden_dim)
        self.q2_out_layer = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.q1_base(xu), inplace=False)
        x1 = self.q1_out_layer(x1)
        x2 = F.relu(self.q2_base(xu), inplace=False)
        x2 = self.q2_out_layer(x2)

        return x1, x2

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6

class MujocoPolicy(BaseNetwork): # MLP base and Gassian distribution.
    def __init__(self, num_obs, num_actions, action_scale, action_bias, hidden_dim=256):
        super().__init__()

        self.action_scale = action_scale
        self.action_bias = action_bias
        self.base = MLPBase(num_obs, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, num_actions)
        self.log_std_layer = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.base(state)
        mean = self.mean_layer(x)
        logstd = torch.clamp(self.log_std_layer(x), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        action_dist = Normal(mean, logstd.exp())
        return action_dist

    def greedy_action(self, states): # sample actions mainly for trianing Q function.
        with torch.no_grad():
            pi_action_dist = self.forward(states)
        greedy_actions = torch.tanh(pi_action_dist.mean) * self.action_scale + self.action_bias
        return greedy_actions # mean actions

    def sample_random_action(self, states, policy2=None): # sample actions for trianing policy.
        pi1_action_dist = self.forward(states)
        x_t = pi1_action_dist.rsample()
        y_t = torch.tanh(x_t) # 将输出压缩在(-1,1)
        actions = y_t * self.action_scale + self.action_bias
        if policy2 != None:
            with torch.no_grad():
                pi2_action_dist = policy2(states)
            pi1_log_prob = pi1_action_dist.log_prob(x_t).sum(-1, keepdim=True)- torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon).sum(1, keepdim=True)
            pi2_log_prob = pi2_action_dist.log_prob(x_t).sum(-1, keepdim=True)- torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon).sum(1, keepdim=True)
            entropies = -pi1_log_prob.mean()
            cross_entropies = -pi2_log_prob.mean()
            return actions, entropies, cross_entropies
        else:
            return actions # random actions

