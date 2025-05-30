import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# --- 辅助网络定义 ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # 输出 logits
        )

    def forward(self, state):
        return self.network(state)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出状态值 V(s)
        )

    def forward(self, state):
        return self.network(state)

# --- QAC 算法实现 ---
class QAC:
    def __init__(self, state_dim, action_dim, actor_learning_rate=0.001, critic_learning_rate=0.001, gamma=0.99,
                 device='cpu'):  # Added device
        self.gamma = gamma
        self.device = device  # Store device
        self.actor = PolicyNetwork(state_dim, action_dim).to(self.device)  # Move actor to device
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic = ValueNetwork(state_dim).to(self.device)  # Move critic to device
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # Move state_tensor to device
        action_logits = self.actor(state_tensor)  # logits on device
        m = Categorical(logits=action_logits)
        action = m.sample()  # action on device
        return action.item()  # Return Python int

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float).to(self.device)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        done_tensor = torch.tensor([done], dtype=torch.float).to(self.device)

        current_v = self.critic(state_tensor)  # on device
        with torch.no_grad():
            next_v = self.critic(next_state_tensor)  # on device
            td_target = reward_tensor + self.gamma * next_v * (1 - done_tensor)  # on device

        # Ensure td_target is properly detached for critic loss calculation if it involved gradients
        # For MSE loss, target should be detached if it comes from a network that might require_grad
        # Here, next_v is from no_grad block, so td_target won't carry grads from next_v path.
        critic_loss = F.mse_loss(current_v, td_target.detach())  # on device
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Detach advantage from critic's computation graph
        advantage = (td_target - current_v).detach()  # on device

        action_logits = self.actor(state_tensor)  # on device
        m = Categorical(logits=action_logits)
        log_prob = m.log_prob(action_tensor)  # on device

        actor_loss = -(log_prob * advantage).mean()  # on device
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()