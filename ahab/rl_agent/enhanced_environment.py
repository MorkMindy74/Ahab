# file: ahab/rl_agent/enhanced_environment.py
"""
EnhancedPortfolioEnv — a drop-in replacement for the original PortfolioEnv with:

1. RISK-AWARE REWARD  (Sortino Ratio) instead of raw return.
2. AUTOMATIC STOP-LOSS: if drawdown from peak exceeds `stop_loss_threshold`
   the episode ends immediately (all positions liquidated) with a heavy penalty.
3. SAFE-HARBOR ACTION: an extra action dimension (#n_assets + 1).  If its
   absolute value is the largest in the vector the agent declares "cash out" and
   the episode ends as a hold: no penalty, no bonus, just the current PnL.
4. REALISTIC COSTS: slippage + per-trade commission already in ExecutionHandler.
5. RICHER STATE: uses EnhancedDataHandler to include RSI, MACD and VIX in the
   lookback window (3 features per asset vs 1 before).
"""

import numpy as np
import pandas as pd
import random
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from ..data.enhanced_data_handler import EnhancedDataHandler
from ..portfolio.portfolio_manager import PortfolioManager
from ..execution.execution_handler import ExecutionHandler


class EnhancedPortfolioEnv:
    # ------------------------------------------------------------------ #
    #  Constructor                                                          #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        assets_filepath: str,
        start_date: str,
        end_date: str,
        initial_cash: float = 100_000.0,
        lookback_window: int = 60,
        max_episode_days: int = 60,
        randomize_start: bool = True,
        min_episode_days: int = 250,
        stop_loss_threshold: float = 0.10,   # 10 % drawdown from peak → stop
        risk_free_daily: float = 0.0,        # daily risk-free rate (e.g. 4%/252)
    ):
        self.assets_filepath = assets_filepath
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.lookback_window = lookback_window
        self.max_episode_days = max_episode_days
        self.randomize_start = randomize_start
        self.min_episode_days = min_episode_days
        self.stop_loss_threshold = stop_loss_threshold
        self.risk_free_daily = risk_free_daily

        self.data_handler = EnhancedDataHandler(assets_filepath, start_date, end_date)
        self.symbols = self.data_handler.symbols
        self.n_assets = len(self.symbols)

        self.portfolio_manager = PortfolioManager(self.symbols, self.initial_cash)
        self.execution_handler = ExecutionHandler(
            commission_rate=0.001,   # 0.10 % per trade
            slippage_rate=0.0005,    # 0.05 % slippage
        )

        self.available_dates = self.data_handler.close.index.tolist()
        self.current_step_index = 0

        # +1 for the safe-harbor (cash-out) dimension
        self.action_space_dim = self.n_assets + 1
        self.state_space_dim = self._calc_state_dim()

        # Episode state (initialised properly in reset())
        self.done = True
        self.current_date = None
        self.current_prices = None

    # ------------------------------------------------------------------ #
    #  Gymnasium-ish interface                                              #
    # ------------------------------------------------------------------ #

    def reset(self, episode_start_idx: int = None):
        self.portfolio_manager = PortfolioManager(self.symbols, self.initial_cash)

        # Choose start index
        min_start = self.lookback_window
        max_start = len(self.available_dates) - self.min_episode_days
        if episode_start_idx is not None:
            self.current_step_index = episode_start_idx
        elif self.randomize_start:
            self.current_step_index = random.randint(min_start, max(min_start, max_start))
        else:
            self.current_step_index = min_start

        self.done = False
        self._episode_step = 0

        self._peak_value = self.initial_cash
        self._prev_value = self.initial_cash
        self._last_daily_return = 0.0
        self._daily_returns: list[float] = []
        self._current_step_transactions: list[dict] = []
        self._episode_start_value = self.initial_cash

        self.current_date = self.available_dates[self.current_step_index]
        self.current_prices = self.data_handler.close.loc[self.current_date]
        self.portfolio_manager.record_portfolio_value(self.current_date, self.current_prices)

        return self._get_state()

    def step(self, action_vector: np.ndarray):
        if self.done:
            return self._get_state(), 0.0, True, {}

        self._current_step_transactions = []
        action_vector = np.clip(action_vector, -1.0, 1.0)

        # ---- Safe-harbor check ------------------------------------------------
        # The last element of the action vector is the safe-harbor signal.
        # If |safe_harbor| > max(|asset_actions|) the agent cashes out.
        safe_harbor_signal = action_vector[-1]
        asset_signals = action_vector[:-1]
        if abs(safe_harbor_signal) > np.max(np.abs(asset_signals)) + 0.05:
            reward = self._close_all_positions()
            self.done = True
            info = self._build_info(liquidating=True)
            return self._get_state(), reward, self.done, info

        # ---- Normal trade step ------------------------------------------------
        self._execute_trades(asset_signals)

        # Advance time
        self.current_step_index += 1
        self._episode_step += 1

        if self.current_step_index >= len(self.available_dates):
            self.done = True
        else:
            self.current_date = self.available_dates[self.current_step_index]
            self.current_prices = self.data_handler.close.loc[self.current_date]
            self.portfolio_manager.record_portfolio_value(self.current_date, self.current_prices)

        # Portfolio value after this step
        cur_val = float(self.portfolio_manager.portfolio_value["value"].iloc[-1])
        self._peak_value = max(self._peak_value, cur_val)
        daily_ret = (cur_val / self._prev_value) - 1.0 if self._prev_value > 0 else 0.0
        self._last_daily_return = daily_ret
        self._daily_returns.append(daily_ret)
        self._prev_value = cur_val

        # ---- Stop-loss --------------------------------------------------------
        drawdown_from_peak = 1.0 - cur_val / self._peak_value
        if drawdown_from_peak >= self.stop_loss_threshold:
            penalty = self._close_all_positions(force_penalty=True)
            self.done = True
            return self._get_state(), penalty, self.done, self._build_info()

        # ---- Episode length cap -----------------------------------------------
        if self._episode_step >= self.max_episode_days:
            self.done = True

        # ---- Reward -----------------------------------------------------------
        if self.done:
            reward = self._terminal_reward(cur_val)
        else:
            reward = 0.0   # sparse intermediate reward

        return self._get_state(), reward, self.done, self._build_info()

    # ------------------------------------------------------------------ #
    #  State                                                                #
    # ------------------------------------------------------------------ #

    def _calc_state_dim(self) -> int:
        # 1 (cash ratio) + n_assets (holdings ratio)
        # + n_assets * 3 * lookback  (close+RSI+MACD per asset)
        # + lookback (VIX window)
        return 1 + self.n_assets + (self.n_assets * 3 * self.lookback_window) + self.lookback_window

    def _get_state(self) -> np.ndarray:
        if self.current_date is None:
            return np.zeros(self.state_space_dim, dtype=np.float32)

        pv = self.portfolio_manager.portfolio_value
        total_val = float(pv["value"].iloc[-1]) if not pv.empty else self.initial_cash
        if total_val < 1.0 or not np.isfinite(total_val):
            total_val = 1.0

        cash = self.portfolio_manager.get_cash_balance()
        norm_cash = float(np.clip(cash / total_val, 0, 1))

        holdings = self.portfolio_manager.get_holdings()
        holdings_vals = np.array([
            holdings.get(s, 0) * self._price(s) for s in self.symbols
        ], dtype=np.float32)
        norm_holdings = np.clip(holdings_vals / total_val, 0, 1)

        market_feat = self.data_handler.get_lookback_features(self.current_date, self.lookback_window)

        state = np.concatenate([[norm_cash], norm_holdings, market_feat])
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Reward                                                               #
    # ------------------------------------------------------------------ #

    def _terminal_reward(self, final_value: float) -> float:
        """
        Sortino-based terminal reward.
        Sortino = (mean_daily_return - Rf) / downside_std * sqrt(252)
        We cap it to [-3, 3] to avoid extreme outliers.
        """
        if len(self._daily_returns) < 2:
            return (final_value / self.initial_cash) - 1.0

        ret_arr = np.array(self._daily_returns)
        mean_ret = ret_arr.mean() - self.risk_free_daily
        downside = ret_arr[ret_arr < 0]
        down_std = np.std(downside) if len(downside) > 1 else 1e-9
        sortino = mean_ret / (down_std + 1e-9) * math.sqrt(252)
        return float(np.clip(sortino, -3.0, 3.0))

    def _close_all_positions(self, force_penalty: bool = False) -> float:
        """
        Liquidate all holdings at current prices.
        Returns a terminal reward. If force_penalty (stop-loss triggered),
        adds an extra punishment proportional to the drawdown.
        """
        for sym in self.symbols:
            qty = self.portfolio_manager.holdings.get(sym, 0)
            if qty > 0:
                price = self._price(sym)
                if price:
                    signal = {"symbol": sym, "action": "SELL", "quantity": qty}
                    tx = self.execution_handler.execute_order(signal, self.current_prices)
                    if tx:
                        self.portfolio_manager.update_holdings(sym, -tx["quantity"], tx["price"])
                        self._current_step_transactions.append({
                            "symbol": sym,
                            "action": "SELL",
                            "quantity": tx["quantity"],
                            "price": tx["price"],
                            "cost": tx["cost"],
                        })

        # Recalculate after liquidation
        cur_val = self.portfolio_manager.get_cash_balance()
        prev_val = self._prev_value
        self._last_daily_return = (cur_val / prev_val) - 1.0 if prev_val > 0 else 0.0
        self.portfolio_manager.record_portfolio_value(self.current_date, self.current_prices)
        self._prev_value = cur_val
        pnl_ratio = cur_val / self.initial_cash - 1.0

        if force_penalty:
            # Extra punishment = drawdown amount
            dd = 1.0 - cur_val / self._peak_value
            return float(np.clip(pnl_ratio - dd, -3.0, 3.0))
        else:
            # Voluntary cash-out: just give the PnL
            return float(np.clip(pnl_ratio, -3.0, 3.0))

    # ------------------------------------------------------------------ #
    #  Trade execution                                                      #
    # ------------------------------------------------------------------ #

    def _execute_trades(self, action_vector: np.ndarray):
        THRESHOLD = 0.1
        MAX_POSITION = 0.20        # max 20 % of portfolio in any single asset
        MIN_TRADE_VALUE = 100.0
        CASH_RESERVE = 0.05        # keep 5 % in cash at all times

        sells = [{"symbol": self.symbols[i], "strength": abs(a)}
                 for i, a in enumerate(action_vector) if a < -THRESHOLD]
        buys  = [{"symbol": self.symbols[i], "strength": a}
                 for i, a in enumerate(action_vector) if a > THRESHOLD]

        # --- SELLs first ---
        for sell in sorted(sells, key=lambda x: x["strength"], reverse=True):
            sym = sell["symbol"]
            price = self._price(sym)
            if price is None:
                continue
            qty = self.portfolio_manager.holdings.get(sym, 0) * sell["strength"]
            if qty * price > MIN_TRADE_VALUE:
                sig = {"symbol": sym, "action": "SELL", "quantity": qty}
                tx = self.execution_handler.execute_order(sig, self.current_prices)
                if tx:
                    self.portfolio_manager.update_holdings(sym, -tx["quantity"], tx["price"])
                    self._current_step_transactions.append({
                        "symbol": sym, "action": "SELL",
                        "quantity": tx["quantity"], "price": tx["price"], "cost": tx["cost"]
                    })

        # --- BUYs ---
        total_val = self.portfolio_manager.get_total_portfolio_value(self.current_prices)
        cash_for_buying = self.portfolio_manager.get_cash_balance() - total_val * CASH_RESERVE
        total_buy_strength = sum(b["strength"] for b in buys)

        if total_buy_strength > 0 and cash_for_buying > MIN_TRADE_VALUE:
            spent = 0.0
            for buy in sorted(buys, key=lambda x: x["strength"], reverse=True):
                available = cash_for_buying - spent
                if available <= MIN_TRADE_VALUE:
                    break
                sym = buy["symbol"]
                price = self._price(sym)
                if price is None:
                    continue
                ideal = available * (buy["strength"] / total_buy_strength)
                cur_pos = self.portfolio_manager.holdings.get(sym, 0) * price
                max_add = total_val * MAX_POSITION - cur_pos
                to_spend = min(ideal, max_add)
                if to_spend > MIN_TRADE_VALUE:
                    sig = {"symbol": sym, "action": "BUY", "quantity": to_spend / price}
                    tx = self.execution_handler.execute_order(sig, self.current_prices)
                    if tx:
                        self.portfolio_manager.update_holdings(sym, tx["quantity"], tx["price"])
                        spent += tx["cost"]
                        self._current_step_transactions.append({
                            "symbol": sym, "action": "BUY",
                            "quantity": tx["quantity"], "price": tx["price"], "cost": tx["cost"]
                        })

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _price(self, symbol: str):
        if self.current_prices is None:
            return None
        p = self.current_prices.get(symbol)
        if p is None:
            return None
        val = p.item() if hasattr(p, "item") else float(p)
        return val if pd.notna(val) and val > 1e-6 else None

    def _build_info(self, liquidating: bool = False) -> dict:
        pv = self.portfolio_manager.portfolio_value
        cur_val = float(pv["value"].iloc[-1]) if not pv.empty else self.initial_cash
        return {
            "final_portfolio_value": cur_val,
            "current_date": self.current_date,
            "daily_return_ratio": self._last_daily_return,
            "cash": self.portfolio_manager.get_cash_balance(),
            "assets_value": cur_val - self.portfolio_manager.get_cash_balance(),
            "transactions": self._current_step_transactions.copy(),
            "liquidating": liquidating,
        }
