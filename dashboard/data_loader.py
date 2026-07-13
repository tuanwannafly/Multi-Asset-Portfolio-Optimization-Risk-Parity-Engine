"""Data loader for the backtest dashboard.

Reads pre-computed artefacts from ``results/`` produced by ``run.py`` and
caches them on the Streamlit side so the dashboard stays snappy.

The module is deliberately defensive: every loader handles a missing or
malformed file by returning ``None`` (or a sensible empty default) rather
than raising, so the UI can degrade gracefully on partial runs.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


METHOD_LABELS = {
    "equal_weight": "Equal Weight (1/N)",
    "mean_variance": "Mean-Variance",
    "risk_parity": "Risk Parity",
    "hrp": "Hierarchical Risk Parity",
    "black_litterman": "Black-Litterman",
}

METHOD_ORDER = list(METHOD_LABELS.keys())


def _repo_root() -> Path:
    """Locate the repo root whether we run from app.py or dashboard/app.py.

    HF Spaces runs the app from the repo root, so ``Path(__file__).parent.parent``
    is the parent of ``dashboard/`` and equals the repo root.
    """
    return Path(__file__).resolve().parent.parent


def results_dir() -> Path:
    return _repo_root() / "results"


# ---------------------------------------------------------------------------
# CSV / JSON readers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_equity_curves() -> pd.DataFrame:
    """Daily NAV per method (date index)."""
    path = results_dir() / "equity_curves.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index.name = "date"
    return df.sort_index()


@st.cache_data(show_spinner=False)
def load_portfolio_returns() -> pd.DataFrame:
    """Daily arithmetic returns per method (date index)."""
    path = results_dir() / "portfolio_returns.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index.name = "date"
    return df.sort_index()


@st.cache_data(show_spinner=False)
def load_comparison_table() -> pd.DataFrame:
    """Method-level metric table (Sharpe, Sortino, Calmar, ...)."""
    path = results_dir() / "comparison_table.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index.name = "method"
    # Keep stable order
    df = df.reindex([m for m in METHOD_ORDER if m in df.index])
    return df


@st.cache_data(show_spinner=False)
def load_weights(method: str) -> pd.DataFrame:
    """Per-rebalance weight history for one method (date index, asset columns)."""
    path = results_dir() / f"weights_{method}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index.name = "date"
    return df.sort_index()


@st.cache_data(show_spinner=False)
def load_diagnostics(method: str) -> pd.DataFrame:
    """Per-rebalance diagnostics: turnover, cost, exposure, realized vol, DD."""
    path = results_dir() / f"diagnostics_{method}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index.name = "date"
    return df.sort_index()


@st.cache_data(show_spinner=False)
def load_robustness_sweep() -> pd.DataFrame:
    path = results_dir() / "robustness_sweep.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_robustness_summary() -> pd.DataFrame:
    path = results_dir() / "robustness_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index.name = "method"
    return df.reindex([m for m in METHOD_ORDER if m in df.index])


@st.cache_data(show_spinner=False)
def load_cost_sensitivity() -> pd.DataFrame:
    path = results_dir() / "cost_sensitivity_sharpe.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index.name = "cost_bps_roundtrip"
    return df


@st.cache_data(show_spinner=False)
def load_sharpe_significance() -> pd.DataFrame:
    path = results_dir() / "sharpe_significance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


@st.cache_data(show_spinner=False)
def load_stress_2022() -> pd.DataFrame:
    path = results_dir() / "stress_test_2022.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index.name = "method"
    return df.reindex([m for m in METHOD_ORDER if m in df.index])


@st.cache_data(show_spinner=False)
def load_efficient_frontier() -> pd.DataFrame:
    path = results_dir() / "efficient_frontier.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index.name = "point"
    return df


@st.cache_data(show_spinner=False)
def load_turnover_summary() -> dict:
    path = results_dir() / "turnover_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_provenance() -> dict:
    path = results_dir() / "provenance.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_risk_parity_verification() -> dict:
    path = results_dir() / "risk_parity_verification.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def drawdown_series(returns: pd.DataFrame) -> pd.DataFrame:
    """Daily drawdown (negative or zero) for each method's return column."""
    if returns.empty:
        return returns
    cum = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return dd


@st.cache_data(show_spinner=False)
def available_methods() -> list[str]:
    """Methods present in the comparison table, in the canonical order."""
    comp = load_comparison_table()
    return [m for m in METHOD_ORDER if m in comp.index]


@st.cache_data(show_spinner=False)
def date_window(equity: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    if equity.empty:
        ts = pd.Timestamp.today().normalize()
        return ts, ts
    return equity.index.min().to_pydatetime(), equity.index.max().to_pydatetime()


def filter_by_window(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def assemble_corr_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return returns
    return returns.corr()
