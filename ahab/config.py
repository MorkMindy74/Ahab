"""
Ahab configuration loader.

Loads YAML config files and provides typed access to all hyperparameters.
Supports merging a base config with overrides for easy experimentation.

Usage:
    from ahab.config import load_config
    cfg = load_config("configs/enhanced.yaml")
    print(cfg.grpo.group_size)  # 16
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── Dataclass hierarchy ──────────────────────────────────────────────────────


@dataclass
class DataConfig:
    assets_file: str = "ahab/assets.txt"
    train_start: str = "2004-01-01"
    train_end: str = "2020-01-01"
    initial_cash: float = 100_000.0
    lookback_window: int = 60


@dataclass
class GRPOConfig:
    group_size: int = 16
    collection_cycles: int = 4
    update_epochs: int = 20
    eps_clip: float = 0.2
    gamma: float = 0.99
    beta_kl: float = 0.01


@dataclass
class ActorConfig:
    lr: float = 1e-5
    action_std_init: float = 0.6
    action_std_min: float = 0.4
    std_decay_rate: float = 0.05
    grad_clip: float = 0.5


@dataclass
class TrainingConfig:
    max_timesteps: int = 3_000_000
    save_freq: int = 50_000
    log_freq: int = 1
    log_dir: str = "logs"
    model_dir: str = "checkpoints"


@dataclass
class ExecutionConfig:
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005


@dataclass
class TradingConfig:
    action_threshold: float = 0.1
    max_position_size: float = 0.20
    min_trade_value: float = 100.0
    cash_reserve: float = 0.05


@dataclass
class RiskConfig:
    stop_loss_threshold: float = 0.10
    risk_free_daily: float = 0.0


@dataclass
class AhabConfig:
    """Top-level configuration container for Ahab."""

    data: DataConfig = field(default_factory=DataConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


# ── Loader ───────────────────────────────────────────────────────────────────

_SECTION_MAP = {
    "data": DataConfig,
    "grpo": GRPOConfig,
    "actor": ActorConfig,
    "training": TrainingConfig,
    "execution": ExecutionConfig,
    "trading": TradingConfig,
    "risk": RiskConfig,
}


def load_config(path: str | Path | None = None) -> AhabConfig:
    """
    Load an Ahab configuration from a YAML file.

    If *path* is ``None``, returns the default configuration.
    """
    if path is None:
        return AhabConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    kwargs: dict[str, Any] = {}
    for section_name, dc_cls in _SECTION_MAP.items():
        section_data = raw.get(section_name, {})
        if section_data:
            kwargs[section_name] = dc_cls(**section_data)

    return AhabConfig(**kwargs)
