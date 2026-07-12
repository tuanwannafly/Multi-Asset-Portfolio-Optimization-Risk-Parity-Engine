"""Portfolio construction methods, exposed as weight functions for the backtest.

Each method follows the signature:
    weight_fn(returns_window, cov_window, cfg) → pd.Series of weights

`returns_window` is the recent log-returns window, `cov_window` the covariance
already estimated (Ledoit-Wolf or sample), and `cfg` is the RebalanceConfig.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.backtest.rebalance_engine import RebalanceConfig
from src.optimizers.mean_variance import mvo_min_variance
from src.optimizers.risk_parity import risk_parity
from src.optimizers.hrp import hrp


def equal_weight(returns: pd.DataFrame, cov: pd.DataFrame, cfg: RebalanceConfig) -> pd.Series:
    """Naive 1/N baseline."""
    w = pd.Series(np.ones(len(returns.columns)), index=returns.columns)
    return w / w.sum()


def mean_variance(returns: pd.DataFrame, cov: pd.DataFrame, cfg: RebalanceConfig) -> pd.Series:
    """MVO with target return constraint (or min-variance if target is None)."""
    mu = returns.mean() * 252.0
    w = mvo_min_variance(
        cov.values,
        mu=mu.values,
        target_return=cfg.target_return_annual,
        long_only=cfg.long_only,
        max_weight=cfg.max_weight,
        min_weight=cfg.min_weight,
    )
    return pd.Series(w, index=returns.columns)


def risk_parity_weights(returns: pd.DataFrame, cov: pd.DataFrame, cfg: RebalanceConfig) -> pd.Series:
    """Equal Risk Contribution weights with max-weight cap."""
    w = risk_parity(
        cov.values,
        long_only=cfg.long_only,
        max_weight=cfg.max_weight,
    )
    return pd.Series(w, index=returns.columns)


def hrp_weights(returns: pd.DataFrame, cov: pd.DataFrame, cfg: RebalanceConfig) -> pd.Series:
    """Hierarchical Risk Parity weights."""
    return hrp(cov, returns=returns)


def black_litterman_weights_fn(
    views: list,
    view_returns: list,
    confidences: list | None = None,
    tau: float = 0.05,
):
    """Closure that builds a BL weight function bound to specific views."""
    from src.optimizers.black_litterman import black_litterman_weights

    def _fn(returns: pd.DataFrame, cov: pd.DataFrame, cfg: RebalanceConfig) -> pd.Series:
        market_w = np.ones(len(returns.columns)) / len(returns.columns)
        return black_litterman_weights(
            cov,
            market_weights=market_w,
            views=views,
            view_returns=view_returns,
            tau=tau,
            confidences=confidences,
            long_only=cfg.long_only,
            max_weight=cfg.max_weight,
        )

    return _fn


METHOD_REGISTRY: Dict[str, callable] = {
    "equal_weight": equal_weight,
    "mean_variance": mean_variance,
    "risk_parity": risk_parity_weights,
    "hrp": hrp_weights,
}


__all__ = list(METHOD_REGISTRY.keys()) + ["black_litterman_weights_fn", "METHOD_REGISTRY"]