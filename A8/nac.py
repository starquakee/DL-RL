import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from aac import A2C





def get_flat_grads_from_model(model):  # Assumes .grad is already populated
    return torch.cat([param.grad.data.view(-1) for param in model.parameters() if param.grad is not None])


class NaturalAC(A2C):  # A2C already handles device for actor/critic and buffers
    def __init__(self, state_dim, action_dim, actor_learning_rate=0.001, critic_learning_rate=0.001,
                 gamma=0.99, gae_lambda=0.95,
                 cg_iters=10, cg_damping=1e-3,
                 fim_update_interval=1, max_kl_step=0.01, device='cpu'):  # Added device

        # Pass device to A2C's __init__
        super().__init__(state_dim, action_dim, actor_learning_rate, critic_learning_rate,
                         gamma, gae_lambda, device=device)
        # self.device is already set by A2C's __init__

        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.fim_update_interval = fim_update_interval
        self._update_counter = 0
        self.param_dim = sum(p.numel() for p in self.actor.parameters())
        # self.actor_device is already self.device
        self.fisher_matrix = torch.zeros((self.param_dim, self.param_dim), device=self.device)  # FIM on device

        self.max_kl_step = max_kl_step
        if hasattr(self, 'actor_optimizer'):  # NPG handles actor updates manually
            del self.actor_optimizer


    def _update_fisher_matrix_approximation(self, states_batch_tensor, actions_batch_tensor):
        self.fisher_matrix.zero_()
        num_samples = states_batch_tensor.size(0)
        if num_samples == 0: return

        for i in range(num_samples):
            state = states_batch_tensor[i:i + 1]
            action = actions_batch_tensor[i:i + 1].long()

            action_logits = self.actor(state)
            dist = Categorical(logits=action_logits)
            log_prob = dist.log_prob(action.squeeze(-1))

            self.actor.zero_grad()
            grad_log_prob_tuple = torch.autograd.grad(log_prob, self.actor.parameters(), retain_graph=False)
            # grad_log_prob_tuple items are on device
            grad_log_prob_flat = torch.cat([g.contiguous().view(-1) for g in grad_log_prob_tuple])

            self.fisher_matrix += torch.outer(grad_log_prob_flat, grad_log_prob_flat)

        if num_samples > 0: self.fisher_matrix /= num_samples

    def _conjugate_gradient_solve(self, vanilla_gradients_flat):
        # vanilla_gradients_flat is on self.device
        F_damped = self.fisher_matrix + self.cg_damping * torch.eye(self.param_dim, device=self.device)

        x_solution = torch.zeros_like(vanilla_gradients_flat, device=self.device)
        residual = vanilla_gradients_flat.clone()
        p_direction = residual.clone()
        rs_old_sq = torch.dot(residual, residual)  # scalar, on device

        for _ in range(self.cg_iters):
            if torch.sqrt(rs_old_sq) < 1e-10: break

            Ap = F_damped @ p_direction
            alpha = rs_old_sq / (torch.dot(p_direction, Ap) + 1e-12)

            x_solution += alpha * p_direction
            residual -= alpha * Ap
            rs_new_sq = torch.dot(residual, residual)
            beta = rs_new_sq / (rs_old_sq + 1e-12)
            p_direction = residual + beta * p_direction
            rs_old_sq = rs_new_sq

        return x_solution

    def _calculate_kl_divergence_for_actor(self, old_actor_logits_detached, states_batch_tensor):
        # inputs are on self.device
        current_actor_logits = self.actor(states_batch_tensor)

        p_old = F.softmax(old_actor_logits_detached, dim=-1)
        log_p_old = F.log_softmax(old_actor_logits_detached, dim=-1)
        log_p_current = F.log_softmax(current_actor_logits, dim=-1)

        kl_per_state = (p_old * (log_p_old - log_p_current)).sum(dim=-1)
        return kl_per_state.mean()

    def update(self, last_next_state_if_not_terminal):  # Overrides A2C's update
        if not self.rewards_buffer: return

        # Data from buffers should already be on self.device due to A2C's store_transition
        rewards_t = torch.stack(self.rewards_buffer).squeeze()
        dones_t = torch.stack(self.dones_buffer).squeeze()
        values_t = torch.stack(self.values_buffer).squeeze()
        states_t_batch = torch.stack(self.states_buffer)
        actions_t_batch = torch.stack(self.actions_buffer).squeeze().long()  # Ensure long for log_prob

        # Calculate value_last_next_state on self.device
        if self.dones_buffer[-1].item():
            value_last_next_state = torch.tensor(0.0, dtype=torch.float, device=self.device)
        else:
            if last_next_state_if_not_terminal is None:
                value_last_next_state = torch.tensor(0.0, dtype=torch.float, device=self.device)
            else:
                state_tensor = torch.from_numpy(last_next_state_if_not_terminal).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    value_last_next_state = self.critic(state_tensor).squeeze()

        values_for_gae = torch.cat((values_t, value_last_next_state.unsqueeze(0)))
        advantages = self.compute_advantage(rewards_t, values_for_gae, dones_t)
        returns_for_critic = advantages.detach() + values_t.detach()

        # Critic Update (same as A2C, already on device)
        current_values_from_critic = self.critic(states_t_batch).squeeze()
        critic_loss = F.mse_loss(current_values_from_critic, returns_for_critic)
        self.critic_optimizer.zero_grad();
        critic_loss.backward();
        self.critic_optimizer.step()

        # Actor Update (Natural Policy Gradient)
        self._update_counter += 1
        if self._update_counter % self.fim_update_interval == 0:
            self._update_fisher_matrix_approximation(states_t_batch.detach(), actions_t_batch.detach())

        action_logits_current = self.actor(states_t_batch)
        dist_current = Categorical(logits=action_logits_current)
        log_probs_for_objective = dist_current.log_prob(actions_t_batch)
        actor_objective_for_grad = (log_probs_for_objective * advantages.detach()).mean()

        self.actor.zero_grad()
        actor_objective_for_grad.backward(retain_graph=True)
        # get_flat_grads_from_model (global helper) will get grads from self.actor (on device)
        vanilla_ascent_pg_flat = get_flat_grads_from_model(self.actor).detach()

        natural_ascent_direction = self._conjugate_gradient_solve(vanilla_ascent_pg_flat)

        # Ensure FIM product is on device for dot product
        fim_product_term = (self.fisher_matrix @ natural_ascent_direction) + self.cg_damping * natural_ascent_direction
        sFs = torch.dot(natural_ascent_direction, fim_product_term.to(self.device))

        if sFs.item() <= 1e-9:
            beta_step_size = 0.0
        else:
            beta_step_size = torch.sqrt(2 * self.max_kl_step / (sFs + 1e-9))

        old_actor_params_flat = self._get_actor_flat_params().clone()
        with torch.no_grad():
            old_actor_logits_for_kl = self.actor(states_t_batch).detach()

        new_actor_params_flat = old_actor_params_flat + beta_step_size * natural_ascent_direction
        self._set_actor_flat_params(new_actor_params_flat)

        with torch.no_grad():
            kl_div = self._calculate_kl_divergence_for_actor(old_actor_logits_for_kl, states_t_batch)
            if kl_div > self.max_kl_step * 1.5:
                fixed_lr_step = self.hparams.get('actor_learning_rate', 0.001) * natural_ascent_direction
                self._set_actor_flat_params(
                    old_actor_params_flat + fixed_lr_step.to(self.device))

        self.actor.zero_grad()
        self._clear_buffers()