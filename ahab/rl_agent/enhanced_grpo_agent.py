# file: ahab/rl_agent/enhanced_grpo_agent.py
"""
EnhancedGRPOAgent — same training algorithm as GRPOAgent but the actor
network is built to match the EXPANDED state space of EnhancedPortfolioEnv:

  state_dim = 1 + n_assets                        (cash + holdings)
            + n_assets * 3 * lookback_window       (close/RSI/MACD per asset)
            + lookback_window                      (VIX window)

The CNN branch now has 3 * n_assets input channels (one conv for each feature
type: price, RSI, MACD).  Everything else (buffer, GRPO update, KL penalty)
is identical to the original.

Improvements over baseline:
- LayerNorm for training stability
- Orthogonal weight initialization for better gradient flow
- Mini-batch training within K-epoch loop
- Adaptive KL coefficient (auto-adjusts based on observed divergence)
- Correct KL sign: KL(π||π_ref) = E[log π - log π_ref]
- Independent(Normal) instead of MultivariateNormal for efficiency
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

# ────────────────────────────────────────────────────────────────────────────
#  Utility activations
# ────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# ────────────────────────────────────────────────────────────────────────────
#  Actor Network
# ────────────────────────────────────────────────────────────────────────────

class EnhancedActorNetwork(nn.Module):
    """
    Hybrid CNN + MLP actor for the enlarged feature space.

    The CNN branch receives a tensor of shape (batch, n_assets * 3, lookback)
    — 3 channels per asset: normalised close, RSI/100, normalised MACD.
    An extra MLP branch handles (cash, holdings, VIX window).

    Includes LayerNorm for training stability and orthogonal weight
    initialization for better gradient flow.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_std_init: float,
        device,
        n_assets: int = 30,
        lookback_window: int = 60,
        n_feat_per_asset: int = 3,   # close + RSI + MACD
    ):
        super().__init__()
        self.device = device
        self.n_assets = n_assets
        self.lookback = lookback_window
        self.n_feat = n_feat_per_asset
        self.action_dim = action_dim

        self.action_var = torch.full(
            (action_dim,), action_std_init ** 2
        ).to(device)

        # ── CNN branch for market sequences ──────────────────────
        in_ch = n_assets * n_feat_per_asset   # e.g. 90 channels
        cnn_out = 256 * lookback_window
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LayerNorm(cnn_out),
        )

        # ── MLP branch for scalars: cash + holdings + VIX window ─
        scalar_dim = 1 + n_assets + lookback_window
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
        )

        # ── Combined actor head ───────────────────────────────────
        combined = cnn_out + 256
        self.actor_head = nn.Sequential(
            nn.Linear(combined, 512 * 2),   # *2 for SwiGLU
            SwiGLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256 * 2),
            SwiGLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
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

    # ── State splitting ─────────────────────────────────────────────────────

    def _split(self, state: torch.Tensor):
        """
        state: (batch, state_dim)

        Layout matches EnhancedPortfolioEnv._get_state():
          [0]          cash ratio
          [1..n]       holdings ratios            → scalar group
          [n+1..n+n*3*L]  close/RSI/MACD flat    → CNN group
          [n+n*3*L+1..]   VIX window              → scalar group
        """
        n, L, F = self.n_assets, self.lookback, self.n_feat
        scalar_end = 1 + n
        market_end = scalar_end + n * F * L
        vix_end    = market_end + L

        scalar  = state[:, :scalar_end]
        market  = state[:, scalar_end:market_end]
        vix_win = state[:, market_end:vix_end]

        market = market.view(-1, n * F, L)
        scalar_full = torch.cat([scalar, vix_win], dim=1)
        return scalar_full, market

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        scalars, market = self._split(state)
        cnn_feat = self.cnn(market)
        mlp_feat = self.mlp(scalars)
        combined = torch.cat([cnn_feat, mlp_feat], dim=1)
        return self.actor_head(combined)

    def set_action_std(self, std: float):
        self.action_var = torch.full(
            (self.action_dim,), std ** 2
        ).to(self.device)

    def _build_dist(self, mean: torch.Tensor):
        std = torch.sqrt(self.action_var).expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def act(self, state: torch.Tensor, deterministic: bool = False):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        mean = self.forward(state)
        if deterministic:
            return mean.detach(), None
        dist = self._build_dist(mean)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.detach(), logp.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        mean = self.forward(state)
        dist = self._build_dist(mean)
        logp = dist.log_prob(action)
        ent  = dist.entropy()
        return logp, ent


# ────────────────────────────────────────────────────────────────────────────
#  Buffer  (unchanged from original)
# ────────────────────────────────────────────────────────────────────────────

class GRPOBuffer:
    def __init__(self):
        self.clear()

    def store_group_episode(self, states, actions, logprobs, rewards, advantages):
        self.states.extend(states)
        self.actions.extend(actions)
        self.logprobs.extend(logprobs)
        self.rewards.extend(rewards)
        self.advantages.extend(advantages)

    def get_batch(self):
        if not self.states:
            return None, None, None, None
        states     = torch.stack(self.states)
        actions    = torch.stack(self.actions)
        logprobs   = torch.stack(self.logprobs).squeeze(-1)
        advantages = torch.tensor(self.advantages, dtype=torch.float32)
        return states, actions, logprobs, advantages

    def clear(self):
        self.states = []; self.actions = []; self.logprobs = []
        self.rewards = []; self.advantages = []


# ────────────────────────────────────────────────────────────────────────────
#  Agent
# ────────────────────────────────────────────────────────────────────────────

class EnhancedGRPOAgent:
    """
    GRPO training logic with the enhanced actor network.

    Improvements:
    - Mini-batch training within K-epoch loop for better gradient estimates
    - Adaptive KL coefficient that auto-adjusts based on observed divergence
    - Configurable entropy coefficient
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float,
        gamma: float,
        K_epochs: int,
        eps_clip: float,
        action_std_init: float,
        device,
        beta_kl: float = 0.01,
        entropy_coef: float = 0.01,
        mini_batch_size: int = 256,
        kl_target: float = 0.01,
        n_assets: int = 30,
        lookback_window: int = 60,
    ):
        self.device      = device
        self.action_std  = action_std_init
        self.gamma       = gamma
        self.eps_clip    = eps_clip
        self.K_epochs    = K_epochs
        self.beta_kl     = beta_kl
        self.entropy_coef = entropy_coef
        self.mini_batch_size = mini_batch_size
        self.kl_target   = kl_target

        self.buffer = GRPOBuffer()

        net_kwargs = dict(
            state_dim=state_dim, action_dim=action_dim,
            action_std_init=action_std_init, device=device,
            n_assets=n_assets, lookback_window=lookback_window,
        )
        self.policy = EnhancedActorNetwork(**net_kwargs).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.reference_policy = EnhancedActorNetwork(**net_kwargs).to(device)
        self.reference_policy.load_state_dict(self.policy.state_dict())
        for p in self.reference_policy.parameters():
            p.requires_grad = False

    # ---- Public API -----------------------------------------------------------

    def set_action_std(self, std: float):
        self.action_std = std
        self.policy.set_action_std(std)

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            s = torch.FloatTensor(state).to(self.device)
            a, lp = self.policy.act(s, deterministic)
        if lp is not None:
            return a.cpu().numpy().flatten(), lp.cpu().item()
        return a.cpu().numpy().flatten(), None

    def generate_group_actions(self, state: np.ndarray, group_size: int):
        out = []
        with torch.no_grad():
            s = torch.FloatTensor(state).to(self.device)
            for _ in range(group_size):
                a, lp = self.policy.act(s, deterministic=False)
                out.append((a.cpu().numpy().flatten(),
                            lp.cpu().item() if lp is not None else None))
        return out

    def calculate_group_advantages(self, group_rewards, normalize: bool = True):
        arr = np.array(group_rewards)
        if normalize and len(arr) > 1:
            return ((arr - arr.mean()) / (arr.std() + 1e-8)).tolist()
        return (arr - arr.mean()).tolist()

    # ---- Training -------------------------------------------------------------

    def train(self) -> dict:
        if not self.buffer.states:
            return {}

        old_s, old_a, old_lp, adv = self.buffer.get_batch()
        if old_s is None:
            return {}

        old_s, old_a, old_lp, adv = (
            old_s.to(self.device), old_a.to(self.device),
            old_lp.to(self.device), adv.to(self.device),
        )
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        stats = {"policy_loss": [], "kl_divergence": [], "entropy": [], "total_loss": []}
        n_samples = old_s.size(0)

        for _ in range(self.K_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_pl, epoch_kl, epoch_ent, epoch_tl = [], [], [], []

            for start in range(0, n_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_samples)
                idx = perm[start:end]

                mb_s = old_s[idx]
                mb_a = old_a[idx]
                mb_lp = old_lp[idx]
                mb_adv = adv[idx]

                lp, ent = self.policy.evaluate(mb_s, mb_a)
                ratios = torch.exp(lp - mb_lp)

                s1 = ratios * mb_adv
                s2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv
                pl = -torch.min(s1, s2).mean()

                with torch.no_grad():
                    ref_lp, _ = self.reference_policy.evaluate(mb_s, mb_a)
                kl   = torch.mean(lp - ref_lp)  # KL(π||π_ref)
                loss = pl + self.beta_kl * kl - self.entropy_coef * ent.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                epoch_pl.append(pl.item())
                epoch_kl.append(kl.item())
                epoch_ent.append(ent.mean().item())
                epoch_tl.append(loss.item())

            stats["policy_loss"].append(np.mean(epoch_pl))
            stats["kl_divergence"].append(np.mean(epoch_kl))
            stats["entropy"].append(np.mean(epoch_ent))
            stats["total_loss"].append(np.mean(epoch_tl))

        # Adaptive KL penalty: adjust beta_kl based on observed KL
        mean_kl = np.mean(stats["kl_divergence"])
        if mean_kl > self.kl_target * 1.5:
            self.beta_kl = min(self.beta_kl * 2.0, 1.0)
        elif mean_kl < self.kl_target / 1.5:
            self.beta_kl = max(self.beta_kl / 2.0, 1e-4)

        self.buffer.clear()
        return {k: np.mean(v) for k, v in stats.items()}

    # ---- Persistence ----------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "policy_state_dict":    self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "action_std":           self.action_std,
            "beta_kl":              self.beta_kl,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.action_std = ck["action_std"]
        self.policy.set_action_std(self.action_std)
        if "beta_kl" in ck:
            self.beta_kl = ck["beta_kl"]
