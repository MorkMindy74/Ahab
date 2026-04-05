# file: train_enhanced.py
"""
Training script for the Enhanced GRPO Portfolio Agent.

Improvements over the original train_grpo.py:
  - Uses EnhancedPortfolioEnv  (Sortino reward, stop-loss, safe-harbor)
  - Uses EnhancedGRPOAgent     (3-channel CNN: close + RSI + MACD; +VIX)
  - Training window: 2013-01-01 → 2023-12-31
  - Test window (held out): 2024-01-01 → 2025-01-01
  - Saves a checkpoint every 50 k timesteps in  enhanced_models/
  - Writes a CSV training log to  enhanced_logs/

Run:
    python train_enhanced.py
"""

import os
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from ahab.rl_agent.enhanced_grpo_environment import EnhancedSingleStateGRPOEnv
from ahab.rl_agent.enhanced_grpo_agent import EnhancedGRPOAgent

# ── Configuration ──────────────────────────────────────────────────────────

ASSETS_FILE     = "ahab/assets.txt"
TRAIN_START     = "2013-01-01"
TRAIN_END       = "2023-12-31"
INITIAL_CASH    = 100_000.0
STOP_LOSS       = 0.10          # 10 % peak-to-trough → episode ends early

GROUP_SIZE      = 16            # G: parallel trajectories per state
M_CYCLES        = 4             # M: data-collection steps before 1 train call
K_EPOCHS        = 20            # K: gradient updates per training batch

LR_ACTOR        = 1e-5
EPS_CLIP        = 0.2
GAMMA           = 0.99
ACTION_STD_INIT = 0.6
MIN_ACTION_STD  = 0.4
STD_DECAY_RATE  = 0.05
BETA_KL         = 0.01

MAX_TIMESTEPS   = int(3e6)
SAVE_FREQ       = 50_000        # save a checkpoint every N timesteps
LOG_FREQ        = 1             # log every training cycle

LOG_DIR   = "enhanced_logs"
MODEL_DIR = "enhanced_models"

# ── Main ───────────────────────────────────────────────────────────────────

def train():
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Train  : {TRAIN_START}  ->  {TRAIN_END}")
    print(f"G={GROUP_SIZE}  M={M_CYCLES}  K={K_EPOCHS}  lr={LR_ACTOR}")
    print("=" * 80)

    # ── Environment ──────────────────────────────────────────────────────
    env = EnhancedSingleStateGRPOEnv(
        group_size=GROUP_SIZE,
        assets_filepath=ASSETS_FILE,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        initial_cash=INITIAL_CASH,
        stop_loss_threshold=STOP_LOSS,
    )
    env.reset()
    print(f"State dim  : {env.state_space_dim}")
    print(f"Action dim : {env.action_space_dim}  (+1 safe-harbor)")

    # ── Agent ─────────────────────────────────────────────────────────────
    n_assets = len(env.base_env.symbols)
    agent = EnhancedGRPOAgent(
        state_dim=env.state_space_dim,
        action_dim=env.action_space_dim,
        lr_actor=LR_ACTOR,
        gamma=GAMMA,
        K_epochs=K_EPOCHS,
        eps_clip=EPS_CLIP,
        action_std_init=ACTION_STD_INIT,
        device=device,
        beta_kl=BETA_KL,
        n_assets=n_assets,
        lookback_window=env.base_env.lookback_window,
    )

    # ── Logging dirs ─────────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "training_summary.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("cycle,timesteps,mean_reward,max_reward,loss,kl,entropy\n")

    # ── Training loop ────────────────────────────────────────────────────
    time_step        = 0
    training_cycle   = 0
    last_save_at     = 0
    t_start          = time.time()

    while time_step <= MAX_TIMESTEPS:
        training_cycle += 1

        # 1. Sync reference policy
        agent.reference_policy.load_state_dict(agent.policy.state_dict())

        # 2. M-loop: collect data
        cycle_rewards, cycle_max_rewards, cycle_ts = [], [], 0
        for _ in range(M_CYCLES):
            # Advance the base env by one step (get a new "prompt" state)
            action_for_step, _ = agent.select_action(env.base_env._get_state())
            _, _, done, _ = env.base_env.step(action_for_step)
            if done:
                env.base_env.reset()

            stats = env.collect_group_data_into_buffer(agent)
            cycle_rewards.append(stats["group_mean_reward"])
            cycle_max_rewards.append(stats["group_max_reward"])
            cycle_ts += stats["timesteps_this_cycle"]

        time_step += cycle_ts

        # 3. Train on accumulated data
        train_stats = agent.train()

        # 4. Logging
        if training_cycle % LOG_FREQ == 0:
            mean_r = np.mean(cycle_rewards)
            max_r  = np.max(cycle_max_rewards)
            loss   = train_stats.get("policy_loss", 0)
            kl     = train_stats.get("kl_divergence", 0)
            ent    = train_stats.get("entropy", 0)
            elapsed = (time.time() - t_start) / 60

            print(
                f"Cycle {training_cycle:4d} | T={time_step:>9,} | "
                f"MaxR={max_r:+.3f} | MeanR={mean_r:+.3f} | "
                f"KL={kl:.4f} | Loss={loss:.6f} | {elapsed:.1f} min"
            )
            with open(log_path, "a") as f:
                f.write(f"{training_cycle},{time_step},{mean_r},{max_r},{loss},{kl},{ent}\n")

        # 5. Save checkpoint
        if (time_step - last_save_at) >= SAVE_FREQ:
            ckpt = os.path.join(MODEL_DIR, f"enhanced_portfolio_{time_step}.pth")
            agent.save(ckpt)
            print(f"  Saved checkpoint -> {ckpt}")
            last_save_at = time_step

        # 6. Action std annealing
        new_std = max(MIN_ACTION_STD,
                      agent.action_std - (STD_DECAY_RATE / MAX_TIMESTEPS) * cycle_ts)
        agent.set_action_std(new_std)

    # Final save
    final_ckpt = os.path.join(MODEL_DIR, f"enhanced_portfolio_FINAL_{time_step}.pth")
    agent.save(final_ckpt)
    print(f"\nTraining complete. Final model: {final_ckpt}")


if __name__ == "__main__":
    train()
