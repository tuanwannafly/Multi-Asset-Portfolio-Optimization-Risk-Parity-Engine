"""Black-Litterman model.

Combines market-implied equilibrium returns with investor views to produce
a posterior expected-return vector that is then fed into MVO.

Formulation (Idzorek 2005 / Black & Litterman 1992):
    μ_BL = [(τΣ)^{-1} + P' Ω^{-1} P]^{-1} [(τΣ)^{-1} μ_π + P' Ω^{-1} Q]

Where:
    τ       : scalar scaling prior uncertainty (typically 0.025-0.05)
    Σ       : covariance of returns
    P       : (K, N) pick matrix — which assets the views reference
    Q       : (K,) expected return of each view
    Ω       : (K, K) diagonal matrix of view uncertainty
    μ_π     : implied equilibrium returns (we default to equal-weight
              historical mean scaled by Sharpe, or use a views-only mode)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("black_litterman")


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: np.ndarray,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Compute π = δ Σ w_mkt — reverse-engineering equilibrium returns.

    Parameters
    ----------
    cov : (N, N) annualized covariance.
    market_weights : (N,) weights of the market portfolio (sum to 1).
    risk_aversion : δ in the CAPM π = δ Σ w formula.
    """
    w = np.asarray(market_weights, dtype=float)
    w = w / w.sum()
    pi = risk_aversion * cov.values @ w
    return pd.Series(pi, index=cov.index)


def view_uncertainty(
    cov: pd.DataFrame,
    pick: np.ndarray,
    view_returns: np.ndarray,
    tau: float,
    confidences: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Construct the diagonal view-uncertainty matrix Ω.

    If `confidences` are provided (each in [0, 1], higher = more confident),
    we scale Ω accordingly: low confidence → large Ω. Default behaviour
    uses Idzorek's method where Ω_ii = (1 - c_i) * τ * (p_i' Σ p_i).
    """
    cov_arr = cov.values
    base = tau * np.einsum("ij,jk,ik->i", pick, cov_arr, pick)
    if confidences is None:
        return np.diag(base)
    conf = np.clip(np.asarray(confidences, dtype=float), 1e-4, 1.0)
    return np.diag((1.0 - conf) * base)


def black_litterman_posterior(
    cov: pd.DataFrame,
    prior: np.ndarray,
    pick: np.ndarray,
    view_returns: np.ndarray,
    tau: float = 0.05,
    confidences: Optional[Sequence[float]] = None,
) -> pd.Series:
    """Compute the Black-Litterman posterior expected return vector.

    Parameters
    ----------
    cov : annualized covariance.
    prior : (N,) implied equilibrium returns π.
    pick : (K, N) view pick matrix.
    view_returns : (K,) view expected returns Q.
    tau : scalar.
    confidences : optional per-view confidence levels in [0, 1].

    Returns
    -------
    pd.Series
        Posterior expected returns (annualized) indexed by ticker.
    """
    cov_arr = cov.values
    n = cov_arr.shape[0]
    prior = np.asarray(prior, dtype=float)
    pick = np.asarray(pick, dtype=float)
    view_returns = np.asarray(view_returns, dtype=float)
    if pick.ndim == 1:
        pick = pick.reshape(1, -1)
    omega = view_uncertainty(cov, pick, view_returns, tau, confidences)
    inv_tau_sigma = np.linalg.inv(tau * cov_arr)
    inv_omega = np.linalg.inv(omega)
    A = inv_tau_sigma + pick.T @ inv_omega @ pick
    b = inv_tau_sigma @ prior + pick.T @ inv_omega @ view_returns
    post = np.linalg.solve(A, b)
    return pd.Series(post, index=cov.index)


def make_pick_matrix(views: List[Dict], tickers: Sequence[str]) -> np.ndarray:
    """Build the (K, N) pick matrix from a list of view dicts.

    Each view dict must have:
        - `assets`: list of tickers
        - `weights`: list of relative weights within the view (must sum to 1
          if absolute, or be signed for relative views).
    Returns
    -------
    np.ndarray
        Row i: weights on each ticker for view i.
    """
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    P = np.zeros((len(views), n))
    for i, view in enumerate(views):
        for t, w in zip(view["assets"], view["weights"]):
            if t in idx:
                P[i, idx[t]] = w
    return P


def black_litterman_weights(
    cov: pd.DataFrame,
    market_weights: np.ndarray,
    views: List[Dict],
    view_returns: Sequence[float],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    confidences: Optional[Sequence[float]] = None,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> pd.Series:
    """Convenience wrapper: implied returns + BL posterior → MVO weights.

    Reuses MVO's min-variance solver so constraints (long-only, max weight)
    apply consistently across methods.
    """
    from .mean_variance import mvo_min_variance  # local import to avoid cycle

    prior = implied_equilibrium_returns(cov, market_weights, risk_aversion=risk_aversion).values
    pick = make_pick_matrix(views, list(cov.index))
    posterior = black_litterman_posterior(
        cov, prior, pick, view_returns, tau=tau, confidences=confidences
    )
    w = mvo_min_variance(
        cov.values,
        mu=posterior.values,
        target_return=None,
        long_only=long_only,
        max_weight=max_weight,
    )
    return pd.Series(w, index=cov.index)


__all__ = [
    "implied_equilibrium_returns",
    "black_litterman_posterior",
    "make_pick_matrix",
    "black_litterman_weights",
]