"""Risk Parity / Equal Risk Contribution (ERC).

Goal: find weights w such that each asset's contribution to portfolio
variance is proportional to (and, in the equal case, equal to) its
share of total risk.

Risk contribution of asset i is defined as:
    RC_i = w_i * (Σ w)_i / sqrt(w' Σ w)

The ERC problem solves for w > 0 with sum(w) = 1 and RC_i / RC_j = 1
for all i, j. We solve it via a Newton-style fixed-point iteration that
is known to converge fast for this problem class (e.g. Spinu 2013).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("risk_parity")


def _risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute per-asset risk contribution: w_i * (Σw)_i / sqrt(w'Σw)."""
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return np.zeros_like(w)
    sigma = np.sqrt(port_var)
    marginal = cov @ w
    return w * marginal / sigma


def _solve_erc(
    cov: np.ndarray,
    b: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """Iterative algorithm for ERC with target risk budget `b`.

    Uses the cyclical coordinate update from Spinu (2013):
        w_i ← sqrt(b_i * σ²_w / (Σ w)_i)
    Normalize after each cycle. Convergence is usually reached in <50
    iterations for typical portfolios.
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    for it in range(max_iter):
        sigma = float(w @ cov @ w)
        if sigma <= 0:
            return np.ones(n) / n
        sigma_sqrt = np.sqrt(sigma)
        new_w = np.sqrt(np.maximum(b * sigma, 1e-16) / np.maximum(cov @ w, 1e-16))
        s = new_w.sum()
        if s <= 0:
            return np.ones(n) / n
        new_w = new_w / s
        if np.max(np.abs(new_w - w)) < tol:
            return new_w
        w = new_w
    LOGGER.debug("ERC reached max_iter=%d without full convergence (max delta=%.2e)", max_iter, float(np.max(np.abs(new_w - w))))
    return w


def risk_parity(
    cov: pd.DataFrame | np.ndarray,
    risk_budget: Optional[np.ndarray] = None,
    long_only: bool = True,
    max_weight: float = 1.0,
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> np.ndarray:
    """Compute ERC weights.

    Parameters
    ----------
    cov : (N, N) covariance matrix.
    risk_budget : (N,) target contribution shares (default = equal = 1/N).
    long_only, max_weight : constraints.

    Returns
    -------
    np.ndarray
        Weights summing to 1.
    """
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    # Small ridge for numerical stability
    cov = cov + 1e-10 * np.eye(cov.shape[0])
    n = cov.shape[0]
    if risk_budget is None:
        b = np.ones(n) / n
    else:
        b = np.asarray(risk_budget, dtype=float)
        b = np.maximum(b, 1e-12)
        b = b / b.sum()
    w = _solve_erc(cov, b, max_iter=max_iter, tol=tol)
    if long_only:
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 0:
            return np.ones(n) / n
        w = w / s
    if max_weight < 1.0:
        # Iteratively project any overweight back into the cap
        for _ in range(50):
            excess = np.maximum(w - max_weight, 0).sum()
            if excess < 1e-10:
                break
            w = np.minimum(w, max_weight)
            s = w.sum()
            if s <= 0:
                return np.ones(n) / n
            w = w / s
        # After projection, recompute via one ERC pass to nudge toward equal risk
        w = _solve_erc(cov, b, max_iter=50, tol=tol)
        if long_only:
            w = np.clip(w, 0.0, None)
            s = w.sum()
            if s > 0:
                w = w / s
    return w


def verify_risk_contributions(
    w: np.ndarray,
    cov: np.ndarray,
    tol: float = 0.05,
) -> dict:
    """Verify that risk contributions are approximately equal (within `tol`)."""
    rc = _risk_contributions(w, cov)
    rc_share = rc / rc.sum() if rc.sum() > 0 else rc
    target = np.ones_like(w) / len(w)
    max_dev = float(np.max(np.abs(rc_share - target)))
    return {
        "risk_contributions": rc,
        "rc_share": rc_share,
        "target_share": target,
        "max_deviation": max_dev,
        "is_equal_within_tol": max_dev < tol,
    }


__all__ = ["risk_parity", "verify_risk_contributions", "_risk_contributions"]