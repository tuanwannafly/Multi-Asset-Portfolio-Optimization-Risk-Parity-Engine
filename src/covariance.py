"""Covariance estimation.

Two estimators are exposed:
- `sample_covariance`: naive; high variance, can be singular when N ~ T.
- `ledoit_wolf_shrinkage`: shrinks toward a structured target (diagonal by
  default), addressing estimation error. The Ledoit-Wolf formula computes
  the optimal shrinkage intensity analytically, removing the need for
  cross-validation.

We additionally expose `condition_number` so the comparison of "why shrinkage
matters" can be made quantitative.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

LOGGER = logging.getLogger("covariance")


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Standard sample covariance (annualized)."""
    return returns.cov() * 252.0


def _shrink_target(cov: np.ndarray, kind: str) -> np.ndarray:
    """Compute a structured shrinkage target matrix."""
    n = cov.shape[0]
    if kind == "diag":
        return np.diag(np.diag(cov))
    if kind == "identity":
        # mean variance times identity — equivalent to assuming zero correlation
        return np.mean(np.diag(cov)) * np.eye(n)
    if kind == "constant_correlation":
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 0.0)
        avg_corr = corr.sum() / (n * (n - 1))
        target_corr = np.full((n, n), avg_corr)
        np.fill_diagonal(target_corr, 1.0)
        return np.outer(std, std) * target_corr
    raise ValueError(f"Unknown shrinkage target: {kind}")


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame,
    target: str = "diag",
    annualize: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage estimator.

    Uses `sklearn.covariance.LedoitWolf` to obtain the optimal shrinkage
    intensity, then re-expresses the result with a custom target matrix
    so we can control the shrinkage structure (diagonal by default).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns.
    target : {"diag", "identity", "constant_correlation"}
        Shrinkage target structure.
    annualize : bool
        If True, scale by 252 to annualize.

    Returns
    -------
    (cov_df, shrinkage)
        Annualized covariance matrix and the shrinkage intensity used.
    """
    lw = LedoitWolf(store_precision=False).fit(returns.fillna(0.0).values)
    sample = np.cov(returns.fillna(0.0).values, rowvar=False, ddof=1)
    T = _shrink_target(sample, kind=target)
    shrunk = (1.0 - lw.shrinkage_) * sample + lw.shrinkage_ * T
    if annualize:
        shrunk = shrunk * 252.0
    cov_df = pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)
    return cov_df, float(lw.shrinkage_)


def rolling_covariance(
    returns: pd.DataFrame,
    window: int,
    method: str = "ledoit_wolf",
    target: str = "diag",
) -> pd.DataFrame:
    """Compute rolling covariance estimates, stacked long-format.

    Output has MultiIndex (date, ticker) with a single column `cov_diag`
    containing the diagonal of the covariance at that timestamp, plus a
    separate wide `cov` attribute accessed via `.loc[date]`.

    More useful is the lower-level helper below that returns the matrix.
    """
    out = []
    for end in range(window, len(returns) + 1):
        sub = returns.iloc[end - window : end]
        if method == "sample":
            cov = sample_covariance(sub)
        else:
            cov, _ = ledoit_wolf_shrinkage(sub, target=target)
        date = sub.index[-1]
        for t in cov.columns:
            out.append({"date": date, "ticker": t, "variance": cov.loc[t, t]})
    return pd.DataFrame(out).set_index(["date", "ticker"])


def condition_number(cov: pd.DataFrame) -> float:
    """Ratio of largest to smallest eigenvalue — measures ill-conditioning."""
    vals = np.linalg.eigvalsh(cov.values)
    vals = np.abs(vals)
    vals = vals[vals > 1e-12]
    if len(vals) == 0:
        return float("inf")
    return float(vals.max() / vals.min())


def compare_covariance_methods(returns: pd.DataFrame) -> pd.DataFrame:
    """Return a side-by-side comparison of sample vs Ledoit-Wolf covariance.

    Useful for the research note to motivate the choice of shrinkage.
    """
    sample = sample_covariance(returns)
    lw, intensity = ledoit_wolf_shrinkage(returns, target="diag")
    summary = pd.DataFrame(
        {
            "sample_cond": [condition_number(sample)],
            "lw_cond": [condition_number(lw)],
            "lw_shrinkage": [intensity],
            "determinant_sample": [float(np.linalg.det(sample.values))],
            "determinant_lw": [float(np.linalg.det(lw.values))],
        },
        index=["value"],
    ).T
    return summary


if __name__ == "__main__":
    from src.ingestion import fetch_prices, compute_returns
    from src.config_loader import load_config

    cfg = load_config()
    prices = fetch_prices(cfg)
    rets = compute_returns(prices)
    summary = compare_covariance_methods(rets)
    print(summary.round(4))