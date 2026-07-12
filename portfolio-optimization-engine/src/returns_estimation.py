"""Expected return estimation.

Provides rolling historical mean estimation. Designed to be swapped out
for Black-Litterman views or ML-derived signals (e.g. from Project 1).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("returns_estimation")


def historical_mean(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling annualized historical mean return.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns indexed by date.
    window : int
        Lookback window in trading days.

    Returns
    -------
    pd.DataFrame
        Rolling annualized mean; rows where window is incomplete are NaN.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    rolling = returns.rolling(window=window, min_periods=max(60, window // 2)).mean()
    return rolling * 252.0


def exponential_mean(returns: pd.DataFrame, halflife: int = 63) -> pd.DataFrame:
    """Exponentially-weighted annualized mean — more reactive to recent data."""
    weights = returns.ewm(halflife=halflife, min_periods=60).mean()
    return weights * 252.0


def latest_estimate(returns: pd.DataFrame, window: int, method: str = "historical") -> Optional[pd.Series]:
    """Return the most recent expected-return vector (annualized).

    Returns None if no observation has a full window of data.
    """
    if method == "historical":
        mu = historical_mean(returns, window=window)
    elif method == "exponential":
        mu = exponential_mean(returns, halflife=max(20, window // 4))
    else:
        raise ValueError(f"Unknown mean method: {method}")
    valid = mu.dropna(how="all")
    if valid.empty:
        return None
    return valid.iloc[-1]


if __name__ == "__main__":
    # Smoke test on synthetic data
    from src.ingestion import fetch_prices, compute_returns
    from src.config_loader import load_config

    cfg = load_config()
    prices = fetch_prices(cfg)
    rets = compute_returns(prices)
    mu = historical_mean(rets, window=cfg["returns"]["rolling_window"])
    print("Annualized mean return (last row):")
    print(mu.iloc[-1].round(4))