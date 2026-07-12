"""Generate publication-quality charts for the research note.

All charts are saved under `results/`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("viz")

# Consistent palette across all charts
PALETTE = {
    "equal_weight": "#888888",
    "mean_variance": "#d62728",
    "risk_parity": "#1f77b4",
    "hrp": "#2ca02c",
    "black_litterman": "#9467bd",
}


def _colors(methods):
    return [PALETTE.get(m, None) for m in methods]


def plot_equity_curves(equity: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in equity.columns:
        ax.plot(equity.index, equity[col] / equity[col].iloc[0], label=col, color=PALETTE.get(col))
    ax.set_yscale("log")
    ax.set_title("Portfolio Equity Curves (log scale)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


def plot_drawdowns(returns: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in returns.columns:
        equity = (1.0 + returns[col].fillna(0)).cumprod()
        peak = equity.cummax()
        dd = equity / peak - 1.0
        ax.plot(dd.index, dd.values, label=col, color=PALETTE.get(col))
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


def plot_weight_history(weights_dict: Dict[str, pd.DataFrame], out_path: Path, top_n: int = 6) -> None:
    """Stacked area chart of average allocation for top N tickers per method."""
    fig, axes = plt.subplots(len(weights_dict), 1, figsize=(10, 3 * len(weights_dict)), sharex=True)
    if len(weights_dict) == 1:
        axes = [axes]
    for ax, (m, w) in zip(axes, weights_dict.items()):
        if w.empty:
            ax.set_title(f"{m}: no weights")
            continue
        # Use top N tickers by mean weight to keep chart legible
        mean_w = w.mean().sort_values(ascending=False)
        keep = mean_w.head(top_n).index.tolist()
        plot_w = w[keep].copy()
        # Aggregate others into "Other"
        other = w.drop(columns=keep).sum(axis=1)
        plot_w["Other"] = other
        plot_w.plot.area(ax=ax, stacked=True, alpha=0.85, legend=False, colormap="tab20")
        ax.set_title(f"{m} — top {top_n} weights over time")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


def plot_weight_stability(weights_dict: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Bar chart of mean absolute weight change per rebalance — proxy for stability."""
    means = {}
    for m, w in weights_dict.items():
        if w.empty or len(w) < 2:
            continue
        diffs = w.diff().abs().sum(axis=1).dropna()
        means[m] = float(diffs.mean())
    if not means:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    items = sorted(means.items(), key=lambda x: x[1])
    ax.barh([m for m, _ in items], [v for _, v in items], color=[PALETTE.get(m) for m, _ in items])
    ax.set_xlabel("Mean |Δw| per rebalance (lower = more stable)")
    ax.set_title("Weight stability across methods")
    for i, (_, v) in enumerate(items):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


def plot_efficient_frontier(frontier: pd.DataFrame, out_path: Path) -> None:
    if frontier.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frontier["achieved_vol"], frontier["achieved_return"], "o-", color="#1f77b4")
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected return (annualized)")
    ax.set_title("Mean-Variance Efficient Frontier (in-sample)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation matrix (full sample)")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


def plot_dendrogram(Z, labels, out_path: Path) -> None:
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9)
    ax.set_title("Hierarchical clustering of asset correlations")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


__all__ = [
    "plot_equity_curves",
    "plot_drawdowns",
    "plot_weight_history",
    "plot_weight_stability",
    "plot_efficient_frontier",
    "plot_correlation_heatmap",
    "plot_dendrogram",
]