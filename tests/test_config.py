"""Smoke tests for Ahab configuration system."""

from ahab.config import AhabConfig, load_config


def test_default_config():
    cfg = AhabConfig()
    assert cfg.grpo.group_size == 16
    assert cfg.data.initial_cash == 100_000.0
    assert cfg.risk.stop_loss_threshold == 0.10


def test_load_yaml_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.grpo.group_size == 16
    assert cfg.actor.lr == 1e-5
    assert cfg.training.max_timesteps == 3_000_000


def test_load_enhanced_config():
    cfg = load_config("configs/enhanced.yaml")
    assert cfg.data.train_start == "2013-01-01"
    assert cfg.data.train_end == "2023-12-31"
    assert cfg.risk.stop_loss_threshold == 0.10


def test_none_returns_defaults():
    cfg = load_config(None)
    assert isinstance(cfg, AhabConfig)
    assert cfg.grpo.collection_cycles == 4
