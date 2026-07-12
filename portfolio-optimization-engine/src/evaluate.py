"""Portfolio performance evaluation.

Computes standard risk-adjusted metrics plus concentration and turnover
statistics. All metrics are computed on the daily portfolio return series.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return float("nan")
    compounded = (1.0 + returns).prod()
    n = len(returns)
    return compounded ** (periods_per_year / max(n, 1)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return float("nan")
    return float(excess.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    ar = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(ar / abs(mdd))


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """DR = (Σ w_i σ_i) / sqrt(w' Σ w). Higher means more diversified."""
    vols = np.sqrt(np.diag(cov))
    weighted_vol = float(weights @ vols)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    if port_vol == 0:
        return float("nan")
    return weighted_vol / port_vol


def effective_n(weights: np.ndarray) -> float:
    """Effective number of assets = 1 / Σ w_i^2 (inverse Herfindahl)."""
    w = weights[weights > 0]
    if len(w) == 0:
        return 0.0
    return float(1.0 / np.sum(w ** 2))


def summarize(
    returns: pd.Series,
    weights_history: Optional[pd.DataFrame] = None,
    cov: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute summary metrics for a portfolio return series.

    `weights_history` (rebalance_date × ticker) is used to compute average
    effective N. If `cov` is provided, the diversification ratio is reported
    using the latest weights.
    """
    out = {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(returns),
        "max_drawdown": max_drawdown(returns),
    }
    if weights_history is not None and not weights_history.empty:
        avg_n = float(np.mean([effective_n(w.values) for _, w in weights_history.iterrows()]))
        out["effective_n_avg"] = avg_n
        # Weight stability: 1 - mean(|w_t - w_{t-1}|) is harder to read; report mean abs change.
        diffs = weights_history.diff().abs().sum(axis=1).dropna()
        out["mean_weight_change"] = float(diffs.mean()) if not diffs.empty else 0.0
    if cov is not None and weights_history is not None and not weights_history.empty:
        latest_w = weights_history.iloc[-1].values
        out["diversification_ratio"] = diversification_ratio(latest_w, cov.values)
    return out


def comparison_table(
    returns_df: pd.DataFrame,
    weights_dict: dict,
    cov: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a comparison table for all methods.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns (T × method).
    weights_dict : dict
        method → weights history DataFrame.
    cov : optional
        Latest covariance for diversification ratio (computed once for context).
    """
    rows = {}
    for m in returns_df.columns:
        w_hist = weights_dict.get(m)
        rows[m] = summarize(returns_df[m].dropna(), w_hist, cov)
    df = pd.DataFrame(rows).T
    df = df.round(4)
    return df


__all__ = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "diversification_ratio",
    "effective_n",
    "summarize",
    "comparison_table",
]