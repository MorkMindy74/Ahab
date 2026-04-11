"""Ahab — Baseline GRPO training script."""

import os
import time
import pandas as pd
import warnings
import torch
import numpy as np
from ahab.rl_agent.grpo_environment import SingleStateGRPOEnv
from ahab.rl_agent.grpo_agent import GRPOAgent

warnings.filterwarnings("ignore", category=FutureWarning)

def train():
    print("============================================================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("============================================================================================")

    # --- Environment Configuration ---
    assets_filepath = 'ahab/assets.txt'
    start_date = '2004-01-01'
    end_date = '2020-01-01'
    initial_cash = 100000.0
    group_size = 16
    gamma = 0.99
    env = SingleStateGRPOEnv(group_size, assets_filepath, start_date, end_date, initial_cash, gamma=gamma)

    # --- GRPO Agent Hyperparameters ---
    K_epochs = 20
    eps_clip = 0.2
    lr_actor = 1e-5
    action_std_init = 0.6
    beta_kl = 0.01
    min_action_std = 0.60
    action_std_decay_rate = 0.05
    entropy_coef = 0.01
    mini_batch_size = 256
    kl_target = 0.01

    agent = GRPOAgent(
        env.state_space_dim, env.action_space_dim, lr_actor, gamma, K_epochs,
        eps_clip, action_std_init, device, beta_kl,
        entropy_coef=entropy_coef, mini_batch_size=mini_batch_size, kl_target=kl_target,
    )

    # --- Training loop configuration ---
    max_training_timesteps = int(3e6)
    save_model_freq = 50000
    log_freq = 1
    data_collection_cycles_M = 4

    # --- Cosine LR schedule ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        agent.optimizer, T_max=max_training_timesteps // (group_size * data_collection_cycles_M * 60),
        eta_min=lr_actor * 0.1,
    )

    # --- Logging Setup ---
    log_dir, model_dir = "GRPO_logs_SS_M4", "GRPO_models_SS_M4"
    os.makedirs(log_dir, exist_ok=True); os.makedirs(model_dir, exist_ok=True)
    summary_log_path = os.path.join(log_dir, "training_summary.csv")
    if not os.path.exists(summary_log_path):
        with open(summary_log_path, "w") as f:
            f.write("training_cycle,total_timesteps,avg_batch_reward,loss,kl_divergence,entropy,lr,beta_kl\n")

    print("Starting SINGLE-STATE, MULTI-PATH GRPO Training (with M > 1)...")
    print(f"Hyperparameters: Group Size G={group_size}, Collection Cycles M={data_collection_cycles_M}, Update Epochs K={K_epochs}")
    print(f"Mini-batch size={mini_batch_size}, Adaptive KL target={kl_target}, Entropy coef={entropy_coef}")
    print("============================================================================================")

    time_step = 0
    training_update_cycle = 0
    t_start = time.time()
    env.reset()

    while time_step <= max_training_timesteps:

        training_update_cycle += 1

        # 1. UPDATE REFERENCE POLICY
        agent.reference_policy.load_state_dict(agent.policy.state_dict())

        # --- M-Loop (Data Collection) ---
        cycle_group_rewards, cycle_timesteps_collected = [], 0
        cycle_group_max_rewards = []
        for m_step in range(data_collection_cycles_M):
            action_for_step, _ = agent.select_action(env.base_env._get_state())
            _, _, done, _ = env.base_env.step(action_for_step)
            if done: env.base_env.reset()

            group_stats = env.collect_group_data_into_buffer(agent)

            cycle_group_rewards.append(group_stats['group_mean_reward'])
            cycle_group_max_rewards.append(group_stats['group_max_reward'])
            cycle_timesteps_collected += group_stats['timesteps_this_cycle']

        time_step += cycle_timesteps_collected

        # 2. TRAIN ON ACCUMULATED DATA (M*G trajectories)
        training_stats = agent.train()

        # 3. Step LR scheduler
        scheduler.step()

        # 4. LOGGING
        if training_update_cycle % log_freq == 0:
            avg_batch_reward = np.mean(cycle_group_rewards) if cycle_group_rewards else 0
            max_batch_reward = np.max(cycle_group_max_rewards) if cycle_group_max_rewards else 0
            loss = training_stats.get('policy_loss', 0)
            kl = training_stats.get('kl_divergence', 0)
            entropy = training_stats.get('entropy', 0)
            current_lr = scheduler.get_last_lr()[0]
            elapsed = (time.time() - t_start) / 60

            print(
                f"Cycle {training_update_cycle:4d} | T={time_step:>9,} | "
                f"MaxR={max_batch_reward:+.3f} | MeanR={avg_batch_reward:+.3f} | "
                f"KL={kl:.4f} | Loss={loss:.6f} | β_kl={agent.beta_kl:.4f} | "
                f"LR={current_lr:.2e} | {elapsed:.1f} min"
            )

            with open(summary_log_path, "a") as f:
                f.write(f"{training_update_cycle},{time_step},{avg_batch_reward},{loss},{kl},{entropy},{current_lr},{agent.beta_kl}\n")

        # 5. SAVING and DECAY
        if time_step // save_model_freq > (time_step - cycle_timesteps_collected) // save_model_freq:
            print("--------------------------------------------------------------------------------------------")
            print(f"Saving model at timestep {time_step}")
            agent.save(os.path.join(model_dir, f"GRPO_portfolio_{time_step}.pth"))
            print("--------------------------------------------------------------------------------------------")

        agent.set_action_std(max(min_action_std, agent.action_std - (action_std_decay_rate/max_training_timesteps) * cycle_timesteps_collected))

    # Final save
    final_ckpt = os.path.join(model_dir, f"GRPO_portfolio_FINAL_{time_step}.pth")
    agent.save(final_ckpt)
    print(f"\nTraining complete. Final model: {final_ckpt}")

if __name__ == "__main__":
    train()
