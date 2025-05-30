import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class REINFORCE:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, device='cpu'):  # Added device
        self.gamma = gamma
        self.device = device  # Store device
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)  # Move policy_net to device
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # Move state_tensor to device
        probs = self.policy_net(state_tensor)  # probs will be on device
        m = Categorical(probs)
        action = m.sample()  # action will be on device
        log_prob = m.log_prob(action)  # log_prob will be on device
        return action.item(), log_prob  # log_prob remains a tensor

    def store_outcome(self, log_prob, reward):  # log_prob is already a tensor on device
        self.rewards.append(reward)  # reward is a Python number
        self.saved_log_probs.append(log_prob)

    def update(self):
        if not self.rewards: return
        R = 0
        policy_loss = []
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)  # Move returns to device

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        elif len(returns) == 1:  # Handle case with single return
            returns = (returns - returns.mean())  # No division by zero
        # If returns is empty (should be caught by `if not self.rewards`), this block is skipped.

        for log_prob, R_t in zip(self.saved_log_probs, returns):  # log_prob and R_t are on device
            policy_loss.append(-log_prob * R_t)  # policy_loss items are on device

        self.optimizer.zero_grad()
        if policy_loss:
            policy_loss_tensor = torch.stack(policy_loss).sum()  # policy_loss_tensor on device
            policy_loss_tensor.backward()
            self.optimizer.step()
        self.clear_memory()

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_log_probs[:]