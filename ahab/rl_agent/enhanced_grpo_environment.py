# file: ahab/rl_agent/enhanced_grpo_environment.py
"""
EnhancedSingleStateGRPOEnv — wrapper for the "single-state, multi-path" GRPO
paradigm, now using EnhancedPortfolioEnv as the inner environment.

Identical logic to the original SingleStateGRPOEnv, but works with the
expanded state/action space of EnhancedPortfolioEnv.
"""

import copy
import numpy as np
import torch
from .enhanced_environment import EnhancedPortfolioEnv


class EnhancedSingleStateGRPOEnv:

    def __init__(self, group_size: int, assets_filepath: str,
                 start_date: str, end_date: str, initial_cash: float = 100_000.0,
                 stop_loss_threshold: float = 0.10, gamma: float = 0.99):

        self.group_size = group_size
        self.gamma = gamma
        self.env_params = dict(
            assets_filepath=assets_filepath,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            stop_loss_threshold=stop_loss_threshold,
        )
        self.base_env = EnhancedPortfolioEnv(**self.env_params)
        self.state_space_dim  = self.base_env.state_space_dim
        self.action_space_dim = self.base_env.action_space_dim

    def reset(self):
        return self.base_env.reset()

    # ------------------------------------------------------------------ #

    def _run_single_path(self, agent, start_env: EnhancedPortfolioEnv,
                         first_action: np.ndarray) -> dict:
        """Run one full 60-day simulation path from a cloned env state."""
        env = start_env
        states, actions, logprobs, rewards = [], [], [], []

        state = env._get_state()
        with torch.no_grad():
            s_t = torch.FloatTensor(state).to(agent.device)
            a_t = torch.FloatTensor(first_action).to(agent.device)
            lp_t, _ = agent.policy.evaluate(s_t.unsqueeze(0), a_t.unsqueeze(0))
            lp = lp_t.item()

        states.append(state)
        actions.append(first_action)
        logprobs.append(lp)
        next_state, reward, done, info = env.step(first_action)
        rewards.append(reward)

        while not done:
            states.append(next_state)
            action, logprob = agent.select_action(next_state)
            actions.append(action)
            logprobs.append(logprob if logprob is not None else 0.0)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)

        return {
            "states": states, "actions": actions, "logprobs": logprobs,
            "rewards": rewards, "total_reward": sum(rewards),
            "episode_length": len(rewards), "info": info,
        }

    # ------------------------------------------------------------------ #

    def _discount_advantages(self, episode_advantage: float, episode_length: int) -> list:
        """
        Compute per-step discounted advantages from the episode-level advantage.
        Later steps (closer to terminal outcome) receive stronger signal.
        """
        discounted = np.zeros(episode_length)
        for t in range(episode_length):
            steps_to_end = episode_length - 1 - t
            discounted[t] = episode_advantage * (self.gamma ** steps_to_end)
        return discounted.tolist()

    def collect_group_data_into_buffer(self, agent) -> dict:
        """
        GRPO data-collection step:
          1. Generate G diverse first-actions from the current base_env state.
          2. Run G full trajectories (deep-copied envs).
          3. Compute group-relative advantages with per-step discounting.
          4. Push everything into agent.buffer.
        Returns stats dict for logging.
        """
        current_state = self.base_env._get_state()
        group_first_actions, _ = zip(*agent.generate_group_actions(
            current_state, self.group_size
        ))

        group_data = []
        for first_action in group_first_actions:
            cloned = copy.deepcopy(self.base_env)
            ep_data = self._run_single_path(agent, cloned, first_action)
            group_data.append(ep_data)

        group_rewards = [ep["total_reward"] for ep in group_data]
        group_advantages = agent.calculate_group_advantages(group_rewards)

        timesteps = 0
        for ep_data, advantage in zip(group_data, group_advantages):
            ep_len = ep_data["episode_length"]
            timesteps += ep_len
            ep_states   = [torch.FloatTensor(s).to(agent.device) for s in ep_data["states"]]
            ep_actions  = [torch.FloatTensor(a).to(agent.device) for a in ep_data["actions"]]
            ep_logprobs = [torch.FloatTensor([lp]).to(agent.device) for lp in ep_data["logprobs"]]
            ep_advantages = self._discount_advantages(advantage, ep_len)
            agent.buffer.store_group_episode(
                ep_states, ep_actions, ep_logprobs,
                ep_data["rewards"], ep_advantages,
            )

        return {
            "group_mean_reward": np.mean(group_rewards),
            "group_max_reward":  np.max(group_rewards),
            "timesteps_this_cycle": timesteps,
        }
