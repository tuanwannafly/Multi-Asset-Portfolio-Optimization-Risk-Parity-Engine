"""Hierarchical Risk Parity (HRP), following Lopez de Prado (2016).

Three steps:
1. Tree clustering: compute a distance matrix d_ij = sqrt(0.5 * (1 - ρ_ij))
   and apply single-linkage hierarchical clustering.
2. Quasi-diagonalization: reorder columns/rows of the covariance matrix to
   follow the tree's leaf order, producing a quasi-diagonal structure.
3. Recursive bisection: split the ordered assets into two halves at each
   branch, allocate weights inversely proportional to cluster variance.

HRP does not require inverting the covariance matrix, so it remains stable
when sample covariance is ill-conditioned.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform

LOGGER = logging.getLogger("hrp")


def _correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance: d = sqrt(0.5*(1-ρ))."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def hierarchical_cluster(corr: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, list]:
    """Run hierarchical clustering on correlation matrix.

    Returns the linkage matrix and leaf order.
    """
    corr = np.asarray(corr, dtype=float)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    dist = _correlation_to_distance(corr)
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method="single")
    Z = optimal_leaf_ordering(Z, dist_condensed)
    order = leaves_list(Z).tolist()
    return Z, order


def quasi_diagonalize(cov: pd.DataFrame, order: list) -> pd.DataFrame:
    """Reorder covariance so related assets sit next to each other."""
    cov = cov.iloc[order, order]
    return cov


def _cluster_variance(cov: pd.DataFrame) -> pd.Series:
    """Inverse-variance allocation within a cluster of equal weight."""
    diag = np.diag(cov.values)
    inv_var = 1.0 / np.maximum(diag, 1e-12)
    w = inv_var / inv_var.sum()
    return pd.Series(w, index=cov.index)


def _recursive_bisection(cov_ordered: pd.DataFrame) -> pd.Series:
    """Recursive bisection: split cluster into halves, allocate inversely to variance."""
    w = pd.Series(1.0, index=cov_ordered.index)
    items = cov_ordered.index.tolist()

    def bisect(items: list) -> None:
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]
        cov_left = cov_ordered.loc[left, left]
        cov_right = cov_ordered.loc[right, right]
        var_left = _cluster_variance(cov_left).mul(_cluster_variance(cov_left).values, axis=0).sum()
        var_right = _cluster_variance(cov_right).mul(_cluster_variance(cov_right).values, axis=0).sum()
        # Equal-weight cluster variance proxy: use diagonal sum / cluster size
        var_left_proxy = float(np.diag(cov_left.values).sum())
        var_right_proxy = float(np.diag(cov_right.values).sum())
        # Allocate inversely to cluster variance (paraphrasing Lopez de Prado)
        total = var_left_proxy + var_right_proxy
        if total <= 0:
            alloc_left = alloc_right = 0.5
        else:
            alloc_left = 1.0 - var_left_proxy / total
            alloc_right = 1.0 - var_right_proxy / total
            s = alloc_left + alloc_right
            alloc_left /= s
            alloc_right /= s
        for t in left:
            w[t] *= alloc_left
        for t in right:
            w[t] *= alloc_right
        bisect(left)
        bisect(right)

    bisect(items)
    w = w / w.sum()
    return w


def hrp(
    cov: pd.DataFrame | np.ndarray,
    returns: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Compute HRP weights.

    Parameters
    ----------
    cov : (N, N) covariance matrix (DataFrame preferred for labels).
    returns : optional (T, N) returns; used only when cov is a numpy array
        and labels need to be inferred.

    Returns
    -------
    pd.Series
        Weights summing to 1, indexed by ticker.
    """
    if isinstance(cov, np.ndarray):
        if returns is None:
            raise ValueError("Provide either a labeled DataFrame or returns for column names.")
        cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    cov = 0.5 * (cov + cov.T)
    corr = cov.values / np.sqrt(np.outer(np.diag(cov.values), np.diag(cov.values)))
    corr_df = pd.DataFrame(corr, index=cov.index, columns=cov.columns)
    _, order = hierarchical_cluster(corr_df)
    ordered = cov.iloc[order, order]
    weights = _recursive_bisection(ordered)
    weights = weights.reindex(cov.index)
    return weights / weights.sum()


__all__ = ["hrp", "hierarchical_cluster", "quasi_diagonalize"]