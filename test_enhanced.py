# file: test_enhanced.py
"""
Out-of-sample backtest of the Enhanced GRPO agent on 2024-01-01 â†’ 2025-01-01.

Generates:
  enhanced_results/daily_log.csv       â€” portfolio value every trading day
  enhanced_results/transaction_log.csv â€” every trade executed
  enhanced_results/performance.png     â€” 4-panel chart (equity, drawdown,
                                          daily returns, rolling Sortino)

Run:
    python test_enhanced.py
"""

import os
import glob
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import torch
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

from ahab.rl_agent.enhanced_environment import EnhancedPortfolioEnv
from ahab.rl_agent.enhanced_grpo_agent import EnhancedGRPOAgent

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ASSETS_FILE    = "ahab/assets.txt"
TEST_START     = "2024-01-01"
TEST_END       = "2025-01-01"
INITIAL_CASH   = 100_000.0
STOP_LOSS      = 0.10
MODEL_DIR      = "enhanced_models"
OUTPUT_DIR     = "enhanced_results"

# Auto-pick latest checkpoint if MODEL_PATH not set explicitly
MODEL_PATH = None   # â† override here if you want a specific checkpoint

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _find_latest_model(model_dir: str) -> str:
    pths = glob.glob(os.path.join(model_dir, "*.pth"))
    if not pths:
        raise FileNotFoundError(
            f"No .pth files found in '{model_dir}'.\n"
            "Train first:  python train_enhanced.py"
        )
    # Sort by the number embedded in the filename
    def _ts(p):
        import re
        nums = re.findall(r"\d+", os.path.basename(p))
        return int(nums[-1]) if nums else 0
    return max(pths, key=_ts)


def _load_spy_benchmark(start: str, end: str, initial: float) -> pd.Series:
    raw = yf.download("SPY", start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"].squeeze().ffill()
    return (close / close.iloc[0]) * initial


def _sortino(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    excess = returns - rf_daily
    down   = excess[excess < 0]
    down_std = np.std(down) if len(down) > 1 else 1e-9
    return float(excess.mean() / (down_std + 1e-9) * math.sqrt(252))


# â”€â”€ Load agent â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_agent(model_path: str, env: EnhancedPortfolioEnv, device) -> EnhancedGRPOAgent:
    n_assets = len(env.symbols)
    agent = EnhancedGRPOAgent(
        state_dim=env.state_space_dim,
        action_dim=env.action_space_dim,
        lr_actor=0.0,
        gamma=0.99,
        K_epochs=1,
        eps_clip=0.2,
        action_std_init=0.6,
        device=device,
        n_assets=n_assets,
        lookback_window=env.lookback_window,
    )
    agent.load(model_path)
    agent.policy.eval()
    print(f"Model loaded from: {model_path}")
    return agent


# â”€â”€ Run episode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_episode(agent: EnhancedGRPOAgent, env: EnhancedPortfolioEnv, device):
    state = env.reset()
    done  = False
    daily_logs, trx_logs = [], []

    print("Running deterministic test episode...")
    while not done:
        action, _ = agent.select_action(state, deterministic=True)
        state, _, done, info = env.step(action)

        if info:
            daily_logs.append({
                "date":            info["current_date"],
                "portfolio_value": info["final_portfolio_value"],
                "cash":            info["cash"],
                "assets_value":    info["assets_value"],
                "daily_return_pct": info["daily_return_ratio"] * 100,
                "liquidating":     info.get("liquidating", False),
            })
            for tx in info.get("transactions", []):
                trx_logs.append({
                    "date":     info["current_date"],
                    "symbol":   tx["symbol"],
                    "action":   tx["action"],
                    "quantity": tx["quantity"],
                    "price":    tx["price"],
                    "cost":     tx["cost"],
                })

    daily_df = pd.DataFrame(daily_logs)
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.set_index("date")

    trx_df = pd.DataFrame(trx_logs)
    return daily_df, trx_df


# â”€â”€ Analytics & plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def analyse_and_plot(daily_df: pd.DataFrame, trx_df: pd.DataFrame,
                     spy: pd.Series, initial_cash: float):
    if daily_df.empty:
        print("No data to analyse.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # â”€â”€ Save CSVs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    daily_df.to_csv(os.path.join(OUTPUT_DIR, "daily_log.csv"))
    if not trx_df.empty:
        trx_df.to_csv(os.path.join(OUTPUT_DIR, "transaction_log.csv"), index=False)
    print(f"Logs saved to '{OUTPUT_DIR}/'")

    # â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pv         = daily_df["portfolio_value"]
    final_val  = pv.iloc[-1]
    total_ret  = (final_val / initial_cash - 1) * 100

    daily_rets = daily_df["daily_return_pct"].values / 100
    sharpe     = (daily_rets.mean() / (daily_rets.std() + 1e-9)) * math.sqrt(252)
    sortino_v  = _sortino(daily_rets)

    # Drawdown
    cumret  = (1 + daily_df["daily_return_pct"] / 100).cumprod()
    peak    = cumret.expanding().max()
    dd      = (cumret / peak) - 1
    max_dd  = dd.min() * 100

    # SPY metrics
    spy_aligned = spy.reindex(pv.index, method="ffill").ffill()
    spy_ret_pct = (spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1) * 100

    # Rolling 60-day Sortino
    roll_sortino = (
        daily_df["daily_return_pct"]
        .rolling(60)
        .apply(lambda r: _sortino(r / 100), raw=True)
    )

    print("\n" + "=" * 52)
    print(f"  Test period : {pv.index[0].date()} -> {pv.index[-1].date()}")
    print(f"  Initial     : ${initial_cash:>12,.2f}")
    print(f"  Final       : ${final_val:>12,.2f}")
    print(f"  Total Return:  {total_ret:>+.2f}%")
    print(f"  SPY Return  :  {spy_ret_pct:>+.2f}%")
    print(f"  Sharpe      :  {sharpe:>.3f}")
    print(f"  Sortino     :  {sortino_v:>.3f}")
    print(f"  Max Drawdown: {max_dd:>+.2f}%")
    print(f"  Trades      : {len(trx_df)}")
    print("=" * 52 + "\n")

    # â”€â”€ 4-panel plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(4, 1, figsize=(15, 16),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]})
    fig.suptitle(
        f"Enhanced GRPO Agent - Out-of-Sample 2024\n"
        f"Return: {total_ret:+.2f}%   Sortino: {sortino_v:.2f}   "
        f"Max DD: {max_dd:.2f}%",
        fontsize=14,
    )
    fmt_dollar = FuncFormatter(lambda x, _: f"${x:,.0f}")
    fmt_pct    = FuncFormatter(lambda x, _: f"{x:.1f}%")

    # Panel 1 - Equity curves
    ax = axes[0]
    ax.plot(pv.index, pv.values, label="GRPO Agent", color="#2196F3", lw=2)
    ax.plot(spy_aligned.index, spy_aligned.values,
            label="SPY (benchmark)", color="#FF9800", lw=1.5, linestyle="--")
    ax.axhline(initial_cash, color="grey", linestyle=":", lw=1, label="Start")
    ax.stackplot(
        daily_df.index,
        daily_df["cash"],
        daily_df["assets_value"],
        colors=["#B0BEC5", "#4FC3F7"],
        alpha=0.25,
        labels=["Cash", "Assets"],
    )
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Equity Curve vs SPY")
    ax.yaxis.set_major_formatter(fmt_dollar)
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2 - Drawdown
    ax = axes[1]
    ax.fill_between(dd.index, dd * 100, 0, color="#ef5350", alpha=0.5)
    ax.plot(dd.index, dd * 100, color="#b71c1c", lw=1)
    ax.axhline(-STOP_LOSS * 100, color="orange", linestyle="--", lw=1,
               label=f"Stop-loss ({STOP_LOSS*100:.0f}%)")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown from Peak")
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.legend(fontsize=8)

    # Panel 3 - Daily returns bar
    ax = axes[2]
    colours = np.where(daily_df["daily_return_pct"] < 0, "#ef5350", "#66bb6a")
    ax.bar(daily_df.index, daily_df["daily_return_pct"], color=colours, width=1)
    ax.set_ylabel("Daily Return (%)")
    ax.set_title("Daily Returns")
    ax.yaxis.set_major_formatter(fmt_pct)

    # Panel 4 - Rolling 60-day Sortino
    ax = axes[3]
    ax.plot(roll_sortino.index, roll_sortino.values, color="#ab47bc", lw=1.5)
    ax.axhline(1.0, color="grey", linestyle="--", lw=1, label="Sortino = 1")
    ax.axhline(0.0, color="black", linestyle="-", lw=0.5)
    ax.set_ylabel("Sortino (60d)")
    ax.set_title("Rolling 60-Day Sortino Ratio")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(OUTPUT_DIR, "performance.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    model_path = MODEL_PATH or _find_latest_model(MODEL_DIR)

    device = torch.device("cpu")

    env = EnhancedPortfolioEnv(
        assets_filepath=ASSETS_FILE,
        start_date=TEST_START,
        end_date=TEST_END,
        initial_cash=INITIAL_CASH,
        randomize_start=False,
        stop_loss_threshold=STOP_LOSS,
        min_episode_days=1,
    )
    env.max_episode_days = 99_999   # let it run the full year

    agent = load_agent(model_path, env, device)

    daily_df, trx_df = run_episode(agent, env, device)

    spy = _load_spy_benchmark(TEST_START, TEST_END, INITIAL_CASH)

    analyse_and_plot(daily_df, trx_df, spy, INITIAL_CASH)


if __name__ == "__main__":
    main()

