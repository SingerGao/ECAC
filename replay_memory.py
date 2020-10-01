import random
import torch
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, device):
        self.capacity = capacity # 最大容量
        self.buffer = [] # 用list存
        self.position = 0
        self.device = device

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采集样本
        states, actions, rewards, next_states, masks = map(np.stack, zip(*batch))

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)

        return states, actions, rewards, next_states, masks

    def __len__(self):
        return len(self.buffer)
