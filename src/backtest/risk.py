"""Portfolio-level risk overlay: volatility targeting and drawdown control.

Applied *after* weights are computed each rebalance date. Both controls
operate by scaling gross exposure:

    scaled_weights = base_weights * exposure_scalar

Where `exposure_scalar ∈ [0, 1]` combines two factors:
- Drawdown-aware scalar: linearly ramps exposure down from 1.0 at dd=0 to
  0.5 at -max_dd_soft, then to 0.0 at -max_dd_hard. This gives a continuous
  control that recovers as drawdown heals (no permanent kill-switch).
- Vol targeting scalar: clip(target_vol / realized_vol, 0, 1).

The product of the two scalars is the final exposure. When both are at
1.0 the portfolio is fully invested; when both are at 0 the portfolio is
fully defensive.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("risk_overlay")


def realized_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling annualized realized volatility from a portfolio return series."""
    return returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252.0)


def compute_exposure_scalar(
    realized_vol: float,
    target_vol: float,
    drawdown: float,
    max_dd_soft: float = 0.10,
    max_dd_hard: float = 0.20,
    vol_floor: float = 0.01,
) -> float:
    """Return exposure scalar given current vol and drawdown.

    Drawdown-aware scaling (continuous, recovers as drawdown heals):
        - dd in [0, -max_dd_soft]: linear ramp 1.0 → 0.5
        - dd in [-max_dd_soft, -max_dd_hard]: linear ramp 0.5 → 0.0
        - dd ≤ -max_dd_hard: 0.0 (full defensive)

    Vol targeting multiplies the result: clip(target_vol / realized_vol, 0, 1).
    """
    dd = -abs(drawdown) if drawdown is not None else 0.0
    if dd <= -max_dd_hard:
        dd_scalar = 0.0
    elif dd <= -max_dd_soft:
        frac = (-dd - max_dd_soft) / max(max_dd_hard - max_dd_soft, 1e-12)
        dd_scalar = 0.5 * (1.0 - frac)
    elif dd < 0:
        frac = -dd / max(max_dd_soft, 1e-12)
        dd_scalar = 1.0 - 0.5 * frac
    else:
        dd_scalar = 1.0

    if not np.isfinite(realized_vol) or realized_vol < vol_floor:
        vol_scalar = 1.0
    else:
        vol_scalar = float(np.clip(target_vol / realized_vol, 0.0, 1.0))
    return float(dd_scalar * vol_scalar)


def drawdown_path(equity: pd.Series) -> pd.Series:
    """Path of drawdowns (negative numbers) from running peak."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd


def apply_overlay(
    weights: np.ndarray,
    recent_port_returns: pd.Series,
    target_vol: float,
    max_dd_hard: float = 0.20,
    max_dd_soft: float = 0.10,
    vol_lookback: int = 63,
) -> Tuple[np.ndarray, dict]:
    """Scale weights according to risk overlay; return scaled weights and metadata."""
    if recent_port_returns.empty:
        return weights, {
            "exposure_scalar": 1.0,
            "realized_vol": np.nan,
            "drawdown": 0.0,
            "drawdown_breach": False,
        }
    vol = realized_volatility(recent_port_returns, window=vol_lookback).dropna()
    realized_vol = float(vol.iloc[-1]) if not vol.empty else np.nan
    equity = (1.0 + recent_port_returns).cumprod()
    dd = drawdown_path(equity)
    cur_dd = float(dd.iloc[-1])
    scalar = compute_exposure_scalar(
        realized_vol, target_vol, cur_dd, max_dd_soft=max_dd_soft, max_dd_hard=max_dd_hard
    )
    breach = cur_dd <= -abs(max_dd_hard)
    scaled = weights * scalar
    meta = {
        "exposure_scalar": scalar,
        "realized_vol": realized_vol,
        "drawdown": cur_dd,
        "drawdown_breach": breach,
    }
    return scaled, meta


__all__ = [
    "apply_overlay",
    "realized_volatility",
    "drawdown_path",
    "compute_exposure_scalar",
]