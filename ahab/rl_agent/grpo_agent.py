import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
import copy

class SwiGLU(nn.Module):
    """ SwiGLU Activation Function """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class GRPOActorNetwork(nn.Module):
    """
    Actor-only network for GRPO using a hybrid CNN-MLP architecture.
    Includes LayerNorm for training stability and orthogonal weight
    initialization for better gradient flow.
    """
    def __init__(self, state_dim, action_dim, action_std_init, device, n_assets=30, lookback_window=60):
        super(GRPOActorNetwork, self).__init__()

        self.device = device
        self.n_assets = n_assets
        self.lookback_window = lookback_window
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # --- CNN Branch for Market Data ---
        cnn_output_size = 128 * lookback_window
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels=n_assets, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LayerNorm(cnn_output_size),
        )

        # --- MLP Branch for Vector Data (Cash + Holdings) ---
        vector_input_dim = 1 + n_assets
        self.mlp_branch = nn.Sequential(
            nn.Linear(vector_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
        )

        # --- Combined Network with SwiGLU (Actor Head) ---
        combined_dim = cnn_output_size + 128
        self.actor_head = nn.Sequential(
            nn.Linear(combined_dim, 256 * 2),
            SwiGLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256 * 2),
            SwiGLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for linear layers, Xavier for conv layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Final layer should have small weights for stable initial policy
        final_linear = self.actor_head[-2]  # Linear before Tanh
        nn.init.orthogonal_(final_linear.weight, gain=0.01)

    def _split_state(self, state):
        vector_data = state[:, :1 + self.n_assets]
        market_data = state[:, 1 + self.n_assets:]
        market_data = market_data.view(-1, self.n_assets, self.lookback_window)
        return vector_data, market_data

    def forward(self, state):
        vector_data, market_data = self._split_state(state)
        cnn_features = self.cnn_branch(market_data)
        mlp_features = self.mlp_branch(vector_data)
        combined_features = torch.cat((cnn_features, mlp_features), dim=1)
        action_mean = self.actor_head(combined_features)
        return action_mean

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def _build_dist(self, action_mean):
        action_std = torch.sqrt(self.action_var).expand_as(action_mean)
        return Independent(Normal(action_mean, action_std), 1)

    def act(self, state, deterministic=False):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action_mean = self.forward(state)

        if deterministic:
            return action_mean.detach(), None

        dist = self._build_dist(action_mean)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.forward(state)
        dist = self._build_dist(action_mean)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

class GRPOBuffer:
    """
    Buffer for storing group trajectories and advantages for GRPO.
    """
    def __init__(self):
        self.clear()

    def store_group_episode(self, states, actions, logprobs, rewards, group_advantages):
        self.states.extend(states)
        self.actions.extend(actions)
        self.logprobs.extend(logprobs)
        self.rewards.extend(rewards)
        self.advantages.extend(group_advantages)

    def get_batch(self):
        if len(self.states) == 0:
            return None, None, None, None

        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        logprobs = torch.stack(self.logprobs).squeeze(-1)
        advantages = torch.tensor(self.advantages, dtype=torch.float32)

        return states, actions, logprobs, advantages

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.advantages = []

class GRPOAgent:
    """
    GRPO Agent that uses group-relative advantages instead of a critic network.

    Improvements over baseline:
    - Mini-batch training within K-epoch loop for better gradient estimates
    - Adaptive KL coefficient that auto-adjusts based on observed divergence
    - Configurable entropy coefficient
    """
    def __init__(self, state_dim, action_dim, lr_actor, gamma, K_epochs, eps_clip,
                 action_std_init, device, beta_kl=0.01, entropy_coef=0.01,
                 mini_batch_size=256, kl_target=0.01):
        self.device = device
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.beta_kl = beta_kl
        self.entropy_coef = entropy_coef
        self.mini_batch_size = mini_batch_size
        self.kl_target = kl_target  # Target KL for adaptive coefficient

        self.buffer = GRPOBuffer()

        self.policy = GRPOActorNetwork(state_dim, action_dim, action_std_init, device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.reference_policy = GRPOActorNetwork(state_dim, action_dim, action_std_init, device).to(device)
        self.reference_policy.load_state_dict(self.policy.state_dict())
        for param in self.reference_policy.parameters():
            param.requires_grad = False

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy.act(state, deterministic)
        if action_logprob is not None:
            return action.cpu().numpy().flatten(), action_logprob.cpu().item()
        return action.cpu().numpy().flatten(), None

    def select_actions_for_vec(self, states):
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            actions, _ = self.policy.act(states)
        return actions.cpu().numpy()

    def generate_group_actions(self, state, group_size=4):
        actions_and_logprobs = []
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            for _ in range(group_size):
                action, logprob = self.policy.act(state_tensor, deterministic=False)
                actions_and_logprobs.append((
                    action.cpu().numpy().flatten(),
                    logprob.cpu().item() if logprob is not None else None
                ))
        return actions_and_logprobs

    def calculate_group_advantages(self, group_rewards, normalize=True):
        group_rewards = np.array(group_rewards)
        if normalize and len(group_rewards) > 1:
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8
            advantages = (group_rewards - mean_reward) / std_reward
        else:
            mean_reward = np.mean(group_rewards)
            advantages = group_rewards - mean_reward
        return advantages.tolist()

    def train(self):
        """
        Train the policy using GRPO with mini-batch updates and adaptive KL.
        """
        if len(self.buffer.states) == 0:
            return {}

        batch_data = self.buffer.get_batch()
        if batch_data[0] is None:
            return {}

        old_states, old_actions, old_logprobs, advantages = batch_data
        old_states = old_states.to(self.device)
        old_actions = old_actions.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        advantages = advantages.to(self.device)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        training_stats = {'policy_loss': [], 'kl_divergence': [], 'entropy': [], 'total_loss': []}
        n_samples = old_states.size(0)

        for _ in range(self.K_epochs):
            # Mini-batch training: shuffle and iterate over mini-batches
            perm = torch.randperm(n_samples, device=self.device)
            epoch_pl, epoch_kl, epoch_ent, epoch_tl = [], [], [], []

            for start in range(0, n_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_samples)
                idx = perm[start:end]

                mb_states = old_states[idx]
                mb_actions = old_actions[idx]
                mb_old_logprobs = old_logprobs[idx]
                mb_advantages = advantages[idx]

                logprobs, entropy = self.policy.evaluate(mb_states, mb_actions)
                ratios = torch.exp(logprobs - mb_old_logprobs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    ref_logprobs, _ = self.reference_policy.evaluate(mb_states, mb_actions)
                kl_div = torch.mean(logprobs - ref_logprobs)  # KL(π||π_ref)
                total_loss = policy_loss + self.beta_kl * kl_div - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                epoch_pl.append(policy_loss.item())
                epoch_kl.append(kl_div.item())
                epoch_ent.append(entropy.mean().item())
                epoch_tl.append(total_loss.item())

            training_stats['policy_loss'].append(np.mean(epoch_pl))
            training_stats['kl_divergence'].append(np.mean(epoch_kl))
            training_stats['entropy'].append(np.mean(epoch_ent))
            training_stats['total_loss'].append(np.mean(epoch_tl))

        # Adaptive KL penalty: adjust beta_kl based on observed KL
        mean_kl = np.mean(training_stats['kl_divergence'])
        if mean_kl > self.kl_target * 1.5:
            self.beta_kl = min(self.beta_kl * 2.0, 1.0)
        elif mean_kl < self.kl_target / 1.5:
            self.beta_kl = max(self.beta_kl / 2.0, 1e-4)

        self.buffer.clear()
        return {key: np.mean(values) for key, values in training_stats.items()}

    def save(self, checkpoint_path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_std': self.action_std,
            'beta_kl': self.beta_kl,
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.action_std = checkpoint['action_std']
        self.policy.set_action_std(self.action_std)
        if 'beta_kl' in checkpoint:
            self.beta_kl = checkpoint['beta_kl']
