"""Walk-forward rebalancing backtest engine.

Key features:
- Rebalances at a configured frequency (monthly / quarterly).
- Re-estimates expected returns and covariance at each rebalance date using
  ONLY data available at that point (no lookahead).
- Applies transaction costs proportional to turnover (both buy and sell sides).
- Supports per-method weight computation via pluggable `weight_fn`.
- Optional risk overlay (vol-targeting + drawdown kill switch).

Outputs:
- Per-rebalance weight history (T × N × method)
- Daily portfolio return series (T × method)
- Equity curves (T × method)
- Diagnostics: turnover, exposure scalar, realized vol per method
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.covariance import ledoit_wolf_shrinkage, sample_covariance
from src.returns_estimation import historical_mean
from src.backtest.risk import apply_overlay, drawdown_path

LOGGER = logging.getLogger("rebalance_engine")


@dataclass
class RebalanceConfig:
    frequency: str = "monthly"          # monthly | quarterly
    lookback_days: int = 252
    cost_bps: float = 10.0              # combined round-trip cost in bps
    long_only: bool = True
    max_weight: float = 0.30
    min_weight: float = 0.0
    target_return_annual: float = 0.10  # for MVO
    risk_overlay: bool = True
    target_vol_annual: float = 0.12
    vol_lookback_days: int = 63
    max_dd_soft: float = 0.10
    max_dd_hard: float = 0.20
    initial_capital: float = 1_000_000.0


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> List[pd.Timestamp]:
    """Return list of rebalance timestamps (last business day of each period)."""
    if frequency == "monthly":
        groups = index.to_period("M")
    elif frequency == "quarterly":
        groups = index.to_period("Q")
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    df = pd.DataFrame({"ts": index, "group": groups})
    last_per_group = df.groupby("group")["ts"].max().tolist()
    return [pd.Timestamp(t) for t in last_per_group]


def _turnover(old_w: np.ndarray, new_w: np.ndarray) -> float:
    """One-way turnover: 0.5 * Σ |w_new - w_old|.

    Equals sum of buy-side (or equivalently sell-side) absolute trades
    divided by total portfolio value.
    """
    return 0.5 * float(np.sum(np.abs(new_w - old_w)))


def run_backtest(
    returns: pd.DataFrame,
    methods: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, RebalanceConfig], pd.Series]],
    cfg: RebalanceConfig,
    cov_method: str = "ledoit_wolf",
) -> Dict[str, pd.DataFrame]:
    """Walk-forward backtest.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns indexed by date, columns are tickers.
    methods : dict
        Map method_name → callable(returns_window, cov_window, cfg) → pd.Series of weights.
        The signature must accept (returns, cov, cfg) and return weights indexed by ticker.
        For methods that don't need covariance (e.g. equal_weight), the cov argument is ignored.
    cfg : RebalanceConfig
    cov_method : {"ledoit_wolf", "sample"}

    Returns
    -------
    dict
        {
          "equity":            pd.DataFrame (T × method),
          "returns":           pd.DataFrame (T × method),
          "weights":           dict[method → DataFrame (rebalance_date × ticker)],
          "diagnostics":       dict[method → DataFrame with turnover, exposure, vol, dd]
        }
    """
    tickers = list(returns.columns)
    dates = returns.index

    rb_dates = _rebalance_dates(dates, cfg.frequency)
    # First rebalance must allow lookback
    first_valid = dates[cfg.lookback_days - 1] if cfg.lookback_days - 1 < len(dates) else None
    rb_dates = [d for d in rb_dates if d >= first_valid]
    LOGGER.info("Backtest: %d rebalance dates from %s to %s", len(rb_dates), rb_dates[0].date(), rb_dates[-1].date())

    # Initialize state per method
    port_returns = {m: pd.Series(index=dates, dtype=float) for m in methods}
    equity = {m: pd.Series(index=dates, dtype=float) for m in methods}
    weights_hist = {m: [] for m in methods}
    diagnostics = {m: [] for m in methods}
    current_w = {m: pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers) for m in methods}
    nav = {m: cfg.initial_capital for m in methods}

    rb_set = set(rb_dates)

    # Pre-compute index ranges for efficiency
    date_to_pos = {d: i for i, d in enumerate(dates)}

    for i, date in enumerate(dates):
        is_rebalance = date in rb_set
        # Compute new weights at rebalance
        if is_rebalance:
            window_end = date_to_pos[date] + 1
            window_start = max(0, window_end - cfg.lookback_days)
            sub_rets = returns.iloc[window_start:window_end]
            if cov_method == "ledoit_wolf":
                cov, _ = ledoit_wolf_shrinkage(sub_rets, target="diag")
            else:
                cov = sample_covariance(sub_rets)
            # Drop columns that are entirely NaN in the window
            valid = sub_rets.dropna(axis=1, how="all").columns
            sub_rets_valid = sub_rets[valid]
            cov_valid = cov.loc[valid, valid]
            for m, weight_fn in methods.items():
                try:
                    w = weight_fn(sub_rets_valid, cov_valid, cfg)
                    w = w.reindex(tickers).fillna(0.0)
                except Exception as exc:
                    LOGGER.warning("Method %s failed at %s (%s); holding previous weights.", m, date.date(), exc)
                    w = current_w[m].copy()
                # Risk overlay
                if cfg.risk_overlay and i > 0:
                    hist = pd.Series(
                        {d: port_returns[m].iloc[date_to_pos[d]] for d in dates[:i] if not pd.isna(port_returns[m].iloc[date_to_pos[d]])}
                    )
                    w_arr, meta = apply_overlay(
                        w.values,
                        hist,
                        cfg.target_vol_annual,
                        max_dd_hard=cfg.max_dd_hard,
                        max_dd_soft=cfg.max_dd_soft,
                        vol_lookback=cfg.vol_lookback_days,
                    )
                    w_scaled = pd.Series(w_arr, index=tickers)
                    # Normalize raw weights to sum to 1 for storage (so effective_n
                    # is computed on the allocation *structure*, not the exposure scalar)
                    raw_w = w / max(w.sum(), 1e-12)
                else:
                    meta = {"exposure_scalar": 1.0, "realized_vol": np.nan, "drawdown": 0.0, "drawdown_breach": False}
                    w_scaled = w.copy()
                    raw_w = w.copy()
                to = _turnover(current_w[m].values, w_scaled.values)
                cost = to * cfg.cost_bps / 1e4
                nav[m] *= (1.0 - cost)
                diagnostics[m].append({"date": date, "turnover": to, "cost": cost, **meta})
                # Store the *raw* (normalized) weights so concentration metrics reflect
                # the actual allocation choice, not the overlay's exposure scaling.
                weights_hist[m].append({"date": date, **raw_w.to_dict()})
                current_w[m] = w_scaled

        # Apply today's return; if today is a rebalance, charge the
        # transaction cost so it shows up in the return series (and Sharpe).
        for m in methods:
            today_ret = float(returns.iloc[i].fillna(0.0) @ current_w[m].values)
            if is_rebalance and diagnostics[m]:
                today_ret -= diagnostics[m][-1]['cost']
            port_returns[m].iloc[i] = today_ret
            nav[m] *= (1.0 + today_ret)
            equity[m].iloc[i] = nav[m]  # cost already in today_ret

    # Build weight DataFrames
    weights_df = {}
    for m in methods:
        rows = weights_hist[m]
        if not rows:
            weights_df[m] = pd.DataFrame(columns=tickers)
            continue
        df = pd.DataFrame(rows).set_index("date")
        weights_df[m] = df

    diagnostics_df = {}
    for m in methods:
        diagnostics_df[m] = pd.DataFrame(diagnostics[m]).set_index("date") if diagnostics[m] else pd.DataFrame()

    return {
        "equity": pd.DataFrame(equity),
        "returns": pd.DataFrame(port_returns),
        "weights": weights_df,
        "diagnostics": diagnostics_df,
    }


__all__ = ["RebalanceConfig", "run_backtest", "_turnover"]