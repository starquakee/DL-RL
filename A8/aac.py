import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from qac import PolicyNetwork, ValueNetwork

# --- 辅助函数 ---
def get_flat_params_from_model(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def set_flat_params_to_model(model, flat_params):
    offset = 0
    for param in model.parameters():
        param.data.copy_(flat_params[offset:offset + param.numel()].view_as(param.data))
        offset += param.numel()


class A2C:
    def __init__(self, state_dim, action_dim, actor_learning_rate=0.001, critic_learning_rate=0.001,
                 gamma=0.99, gae_lambda=0.95, device='cpu'):  # Added device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device  # Store device

        self.actor = PolicyNetwork(state_dim, action_dim).to(self.device)  # Move to device
        self.critic = ValueNetwork(state_dim).to(self.device)  # Move to device

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.hparams = {'actor_learning_rate': actor_learning_rate}
        self._clear_buffers()

    def _get_actor_flat_params(self):
        # Assuming get_flat_params_from_model is defined globally or accessible
        return get_flat_params_from_model(self.actor)

    def _set_actor_flat_params(self, flat_params):
        # Assuming set_flat_params_to_model is defined globally or accessible
        set_flat_params_to_model(self.actor, flat_params)
    def _clear_buffers(self):  # Buffers store tensors that will be on device
        self.states_buffer = []
        self.actions_buffer = []
        self.log_probs_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []
        self.dones_buffer = []

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # Move to device
        with torch.no_grad():
            action_logits = self.actor(state_tensor)  # on device
            value_tensor = self.critic(state_tensor).squeeze()  # on device, ensure squeezed
        m = Categorical(logits=action_logits)
        action = m.sample()  # on device
        log_prob = m.log_prob(action).squeeze()  # on device, ensure squeezed
        # Return item for action, but tensors for log_prob and value for direct storage
        return action.item(), log_prob, value_tensor

    def store_transition(self, state, action, log_prob_tensor, reward, done, value_tensor):
        self.states_buffer.append(torch.from_numpy(state).float().to(self.device))
        self.actions_buffer.append(torch.tensor(action, dtype=torch.long).to(self.device))
        self.log_probs_buffer.append(log_prob_tensor)  # Already on device
        self.rewards_buffer.append(torch.tensor(reward, dtype=torch.float).to(self.device))
        self.dones_buffer.append(torch.tensor(done, dtype=torch.float).to(self.device))
        self.values_buffer.append(value_tensor)  # Already on device

    def compute_advantage(self, rewards_segment, values_segment_with_last_next, dones_segment):
        # Inputs are expected to be tensors on self.device
        advantages = torch.zeros_like(rewards_segment, device=self.device)
        gae = 0.0
        for t in reversed(range(len(rewards_segment))):
            delta = rewards_segment[t] + \
                    self.gamma * values_segment_with_last_next[t + 1] * (1.0 - dones_segment[t]) - \
                    values_segment_with_last_next[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones_segment[t]) * gae
            advantages[t] = gae
        return advantages

    def update(self, last_next_state_if_not_terminal):
        if not self.rewards_buffer: return

        # Stack tensors from buffers - they are already on self.device
        rewards_t = torch.stack(self.rewards_buffer).squeeze()
        dones_t = torch.stack(self.dones_buffer).squeeze()
        values_t = torch.stack(self.values_buffer).squeeze()  # These are V(S_0) to V(S_{N-1})
        # log_probs_t_old = torch.stack(self.log_probs_buffer).squeeze() # Old log_probs at selection
        states_t_buffer = torch.stack(self.states_buffer)
        actions_t_buffer_from_sel = torch.stack(self.actions_buffer).squeeze()  # Actions that were taken

        if self.dones_buffer[-1].item():  # If trajectory ended with a terminal state
            value_last_next_state = torch.tensor(0.0, dtype=torch.float).to(self.device)
        else:
            if last_next_state_if_not_terminal is None:  # Should not happen if not done
                value_last_next_state = torch.tensor(0.0, dtype=torch.float).to(self.device)
            else:
                state_tensor = torch.from_numpy(last_next_state_if_not_terminal).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    value_last_next_state = self.critic(state_tensor).squeeze()

        # values_for_gae is [V(S_0), ..., V(S_{N-1}), V(S_N)]
        values_for_gae = torch.cat((values_t, value_last_next_state.unsqueeze(0)))

        advantages = self.compute_advantage(rewards_t, values_for_gae, dones_t)  # on device
        returns_for_critic = advantages.detach() + values_t.detach()  # on device

        # Actor update: Get log_probs for *current* policy for actions taken
        action_logits_new = self.actor(states_t_buffer)  # on device
        dist_new = Categorical(logits=action_logits_new)
        # actions_t_buffer_from_sel should be shaped correctly for log_prob
        log_probs_new_for_loss = dist_new.log_prob(actions_t_buffer_from_sel)  # on device

        actor_loss = -(log_probs_new_for_loss * advantages.detach()).mean()  # on device

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        current_values_from_critic = self.critic(states_t_buffer).squeeze()  # on device
        critic_loss = F.mse_loss(current_values_from_critic, returns_for_critic)  # on device
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._clear_buffers()