"""Robustness and sensitivity analyses for the backtest.

Three analyses ship here:

1. ``run_robustness_sweep`` — re-run the backtest over a grid of
   rebalance frequencies and lookback windows. Reports Sharpe / Max DD
   / Turnover for each (method, config) cell.

2. ``run_cost_sensitivity`` — re-run the backtest at multiple
   transaction-cost levels. Standard quant-review question:
   "what happens if costs double?" Answering it requires actually
   re-running, not just subtracting a constant, because cost changes
   re-balance frequency economics.

3. ``summarize_sensitivity`` — pretty-print the sweep results into a
   single table with mean / std / min / max Sharpe per method.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.backtest.rebalance_engine import RebalanceConfig, run_backtest
from src.covariance import ledoit_wolf_shrinkage
from src.evaluate import sharpe_ratio, max_drawdown, annualized_volatility
from src.methods import METHOD_REGISTRY
from src.significance import (
    sharpe_t_stat,
    circular_block_bootstrap_sharpe,
    deflated_sharpe,
)

LOGGER = logging.getLogger("robustness")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_methods(extra_cfg: dict | None = None) -> Dict:
    """Construct the standard method registry, optionally adjusted for cost.

    `extra_cfg` keys we honor:
      - "cost_bps": overrides cfg.cost_bps for this run
    """
    methods = dict(METHOD_REGISTRY)
    # In the main pipeline, BL is constructed with views; for robustness we
    # register the no-views version (momentum-driven views are tested in
    # the main pipeline only).
    try:
        from src.optimizers.black_litterman import black_litterman_weights

        def _bl_no_views(returns, cov, cfg):
            market_w = np.ones(len(returns.columns)) / len(returns.columns)
            w = black_litterman_weights(
                cov,
                market_weights=market_w,
                views=[],
                view_returns=[],
                confidences=None,
                tau=0.05,
                long_only=cfg.long_only,
                max_weight=cfg.max_weight,
            )
            return pd.Series(w, index=returns.columns)

        methods["black_litterman"] = _bl_no_views
    except Exception:
        pass
    return methods


def _summary_metrics(port_returns: pd.DataFrame) -> pd.DataFrame:
    """Quick per-method summary for one backtest run."""
    rows = {}
    for m in port_returns.columns:
        r = port_returns[m].dropna()
        rows[m] = {
            "sharpe": sharpe_ratio(r),
            "ann_vol": annualized_volatility(r),
            "max_dd": max_drawdown(r),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 1) Sweep over rebalance frequency × lookback window
# ---------------------------------------------------------------------------

DEFAULT_FREQ_GRID = ["monthly", "quarterly"]
DEFAULT_LOOKBACK_GRID = [126, 252, 504]  # ~6m, 12m, 24m


def run_robustness_sweep(
    returns: pd.DataFrame,
    freq_grid: Iterable[str] = DEFAULT_FREQ_GRID,
    lookback_grid: Iterable[int] = DEFAULT_LOOKBACK_GRID,
    cost_bps: float = 10.0,
    cov_method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """Run the backtest on a grid of (frequency, lookback) configurations.

    Returns a long DataFrame with one row per (method, frequency, lookback).
    """
    methods = _build_methods()
    rows: List[dict] = []
    for freq in freq_grid:
        for lb in lookback_grid:
            LOGGER.info("Sweep: frequency=%s lookback=%d", freq, lb)
            cfg = RebalanceConfig(
                frequency=freq,
                lookback_days=lb,
                cost_bps=cost_bps,
                long_only=True,
                max_weight=0.30,
                min_weight=0.0,
                target_return_annual=0.12,
                risk_overlay=True,
                target_vol_annual=0.12,
                vol_lookback_days=63,
                max_dd_soft=0.10,
                max_dd_hard=0.20,
                initial_capital=1_000_000.0,
            )
            try:
                bt = run_backtest(returns, methods=methods, cfg=cfg, cov_method=cov_method)
            except Exception as exc:
                LOGGER.warning("Sweep failed for %s/%d (%s)", freq, lb, exc)
                continue
            metrics = _summary_metrics(bt["returns"])
            for m in metrics.index:
                rows.append({
                    "method": m,
                    "frequency": freq,
                    "lookback_days": lb,
                    **metrics.loc[m].to_dict(),
                })
    return pd.DataFrame(rows)


def summarize_sweep(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the sweep into mean / std / min / max per method."""
    if sweep_df.empty:
        return sweep_df
    grouped = sweep_df.groupby("method")["sharpe"]
    out = pd.DataFrame({
        "mean_sharpe": grouped.mean(),
        "std_sharpe": grouped.std(ddof=1),
        "min_sharpe": grouped.min(),
        "max_sharpe": grouped.max(),
        "n_configs": grouped.count(),
    })
    out["range_sharpe"] = out["max_sharpe"] - out["min_sharpe"]
    out = out.round(4)
    return out


# ---------------------------------------------------------------------------
# 2) Cost sensitivity
# ---------------------------------------------------------------------------

DEFAULT_COST_GRID = [0, 10, 25, 50, 100, 200]  # bps round-trip


def run_cost_sensitivity(
    returns: pd.DataFrame,
    cost_grid: Iterable[float] = DEFAULT_COST_GRID,
    frequency: str = "monthly",
    lookback_days: int = 252,
    cov_method: str = "ledoit_wolf",
    risk_overlay: bool = False,
) -> pd.DataFrame:
    """Run the backtest at multiple cost levels (one config; sweep cost only).

    By default ``risk_overlay=False`` so the cost effect shows up cleanly
    (the vol-targeting overlay dampens realized vol and absorbs most of
    the cost). Set ``risk_overlay=True`` to see how the live production
    config degrades under higher costs.

    Returns long DataFrame: (method, cost_bps, sharpe, ann_vol, max_dd).
    """
    methods = _build_methods()
    rows: List[dict] = []
    base_cfg = dict(
        long_only=True,
        max_weight=0.30,
        min_weight=0.0,
        target_return_annual=0.12,
        risk_overlay=risk_overlay,
        target_vol_annual=0.12,
        vol_lookback_days=63,
        max_dd_soft=0.10,
        max_dd_hard=0.20,
        initial_capital=1_000_000.0,
    )
    for cost in cost_grid:
        LOGGER.info("Cost sensitivity: %.0f bps round-trip, overlay=%s", cost, risk_overlay)
        cfg = RebalanceConfig(frequency=frequency, lookback_days=lookback_days,
                              cost_bps=float(cost), **base_cfg)
        try:
            bt = run_backtest(returns, methods=methods, cfg=cfg, cov_method=cov_method)
        except Exception as exc:
            LOGGER.warning("Cost sensitivity failed at %s bps (%s)", cost, exc)
            continue
        metrics = _summary_metrics(bt["returns"])
        for m in metrics.index:
            rows.append({
                "method": m,
                "cost_bps_roundtrip": float(cost),
                **metrics.loc[m].to_dict(),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) Statistical significance for the headline Sharpe
# ---------------------------------------------------------------------------

def run_significance(returns_df: pd.DataFrame, n_trials: int = 1,
                     n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """For each method, compute the IID t-stat and a bootstrap CI for Sharpe."""
    rows = []
    for m in returns_df.columns:
        r = returns_df[m].dropna()
        t_stat, p_val, sr = sharpe_t_stat(r)
        boot = circular_block_bootstrap_sharpe(r, n_boot=n_boot, seed=seed)
        defl = deflated_sharpe(r, n_trials=n_trials)
        rows.append({
            "method": m,
            "sharpe": sr,
            "t_stat_iid": t_stat,
            "p_value_iid": p_val,
            "bootstrap_se": boot["se"],
            "ci_95_low": boot["ci_low"],
            "ci_95_high": boot["ci_high"],
            "block_size": boot["block_size"],
            "n_boot": boot["n_boot"],
            "deflated_sharpe_threshold": defl["deflated_sharpe"],
            "prob_sharpe_above_deflated": defl["prob_sharpe_underestimated"],
            "n_obs": int(len(r)),
        })
    return pd.DataFrame(rows).set_index("method").round(4)


__all__ = [
    "run_robustness_sweep",
    "summarize_sweep",
    "run_cost_sensitivity",
    "run_significance",
    "DEFAULT_FREQ_GRID",
    "DEFAULT_LOOKBACK_GRID",
    "DEFAULT_COST_GRID",
]