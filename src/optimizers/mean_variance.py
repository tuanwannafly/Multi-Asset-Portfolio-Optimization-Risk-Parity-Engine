"""Mean-Variance Optimization (Markowitz) via cvxpy.

Solves: minimize w' Σ w subject to w' μ ≥ target_return, long-only,
max-weight cap. Returns the optimal weight vector.

We additionally expose a utility to sweep across target returns to trace
out the efficient frontier — useful for visualization.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

# Silence cvxpy's verbose output during solve so we don't spam logs.
# SCS prints to stderr by default; redirect via env var before solve.
os.environ.setdefault("SCS_PRINT_LEVEL", "0")

LOGGER = logging.getLogger("mvo")


def _validate_inputs(mu: np.ndarray, cov: np.ndarray) -> None:
    if mu.ndim != 1:
        raise ValueError("mu must be 1D")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    if cov.shape[0] != mu.shape[0]:
        raise ValueError("mu and cov dimension mismatch")


def mvo_min_variance(
    cov: pd.DataFrame | np.ndarray,
    mu: Optional[pd.Series | np.ndarray] = None,
    target_return: Optional[float] = None,
    long_only: bool = True,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> np.ndarray:
    """Solve a constrained mean-variance optimization problem.

    Parameters
    ----------
    cov : (N, N) covariance matrix.
    mu : (N,) expected returns (annualized). Required if `target_return` given.
    target_return : float
        Required portfolio return (annualized). If None, the problem reduces
        to minimum variance without a return target.
    long_only, max_weight, min_weight : portfolio constraints.

    Returns
    -------
    np.ndarray
        Optimal weights summing to 1.
    """
    cov = np.asarray(cov)
    if mu is not None:
        mu = np.asarray(mu, dtype=float)
        _validate_inputs(mu, cov)
    n = cov.shape[0]

    # Symmetrize to avoid numerical issues from non-PSD noise
    cov = 0.5 * (cov + cov.T)
    # Tiny ridge to ensure strict convexity in pathological cases
    cov = cov + 1e-8 * np.eye(n)

    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= min_weight)
    if max_weight < 1.0:
        constraints.append(w <= max_weight)
    if target_return is not None:
        if mu is None:
            raise ValueError("mu must be provided when target_return is set")
        constraints.append(mu @ w >= target_return)

    # Sanity check feasibility of target return. If unreachable given constraints,
    # fall back to the highest-achievable mean return.
    if target_return is not None and mu is not None:
        # Under long-only + max_weight: max w'μ achievable is bounded by
        # sum_{i in top_k} max_weight * μ_i where top_k picks max-weight sized blocks.
        max_achievable = _max_achievable_return(mu, long_only, max_weight, min_weight)
        if target_return > max_achievable + 1e-6:
            LOGGER.debug(
                "Target return %.4f infeasible (max achievable %.4f). Falling back.",
                target_return, max_achievable,
            )
            target_return = max_achievable

    prob = cp.Problem(objective, constraints)
    solved = False
    for solver in (cp.SCS, cp.ECOS, cp.CLARABEL):
        try:
            if solver == cp.SCS:
                prob.solve(solver=solver, verbose=False, max_iters=20000, eps=1e-8)
            else:
                prob.solve(solver=solver, verbose=False)
            # Only accept truly optimal / optimal_inaccurate solutions with finite values
            if w.value is not None and prob.status in (
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            ):
                weights_raw = np.asarray(w.value).flatten()
                if np.all(np.isfinite(weights_raw)):
                    solved = True
                    break
        except (cp.SolverError, Exception):
            continue
    if not solved:
        LOGGER.debug("MVO solver did not converge (status=%s); falling back to equal weights.", prob.status)
        return np.ones(n) / n
    weights = np.clip(np.asarray(w.value).flatten(), 0.0, None)
    if weights.sum() <= 0:
        return np.ones(n) / n
    weights = weights / weights.sum()
    return weights


def _max_achievable_return(mu: np.ndarray, long_only: bool, max_weight: float, min_weight: float) -> float:
    """Upper bound on w'μ under long-only + max_weight constraint."""
    sorted_mu = np.sort(np.asarray(mu, dtype=float))[::-1]
    if not long_only:
        return float(np.sum(np.abs(sorted_mu)))
    n = len(sorted_mu)
    if max_weight >= 1.0:
        return float(np.sum(np.maximum(sorted_mu, 0.0)))
    # Number of slots at max_weight to fill before remainder spread
    k_full = int(1.0 // max_weight)
    if k_full >= n:
        return float(max_weight * np.sum(sorted_mu))
    head = sorted_mu[:k_full]
    rem_slot = 1.0 - k_full * max_weight
    return float(max_weight * np.sum(head) + rem_slot * sorted_mu[k_full])


def efficient_frontier(
    cov: pd.DataFrame,
    mu: pd.Series,
    n_points: int = 25,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """Trace the efficient frontier by sweeping target returns.

    Returns a DataFrame with columns: target_return, achieved_return,
    achieved_vol, sharpe, weights (list).
    """
    cov_arr = np.asarray(cov)
    mu_arr = np.asarray(mu, dtype=float)
    lo = float(np.min(mu_arr))
    hi = float(np.max(mu_arr)) * 1.5
    targets = np.linspace(lo, hi, n_points)
    rows = []
    for t in targets:
        w = mvo_min_variance(cov_arr, mu_arr, target_return=float(t), long_only=long_only, max_weight=max_weight)
        port_ret = float(mu_arr @ w)
        port_vol = float(np.sqrt(w @ cov_arr @ w))
        rows.append({
            "target_return": float(t),
            "achieved_return": port_ret,
            "achieved_vol": port_vol,
            "sharpe": port_ret / port_vol if port_vol > 0 else np.nan,
            "weights": w,
        })
    return pd.DataFrame(rows)


__all__ = ["mvo_min_variance", "efficient_frontier"]