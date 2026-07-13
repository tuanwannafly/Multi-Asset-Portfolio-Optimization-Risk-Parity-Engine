"""Reusable formatting + small plotting helpers for the dashboard."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import data_loader as dl


# Streamlit "Plotly" theme colours used throughout
PALETTE = {
    "equal_weight": "#7f7f7f",      # grey baseline
    "mean_variance": "#1f77b4",     # blue
    "risk_parity": "#2ca02c",       # green
    "hrp": "#9467bd",               # purple
    "black_litterman": "#ff7f0e",   # orange
    "highlight": "#d62728",         # red, used for stress periods
}


def method_color(method: str) -> str:
    return PALETTE.get(method, "#1f77b4")


def method_label(method: str) -> str:
    return dl.METHOD_LABELS.get(method, method)


# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------

def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def fmt_x(x: Optional[float], digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def render_kpi(label: str, value: str, sublabel: Optional[str] = None) -> None:
    st.metric(label=label, value=value, delta=sublabel)


# ---------------------------------------------------------------------------
# Plotly builders
# ---------------------------------------------------------------------------

def equity_chart(equity: pd.DataFrame, methods: list[str]) -> go.Figure:
    fig = go.Figure()
    for m in methods:
        if m not in equity.columns:
            continue
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity[m],
            mode="lines",
            name=method_label(m),
            line=dict(color=method_color(m), width=2),
            hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>NAV: %{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(
        title="Equity curves (start NAV = 1,000,000)",
        xaxis_title="Date",
        yaxis_title="Portfolio value",
        hovermode="x unified",
        template="plotly_white",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def drawdown_chart(dd: pd.DataFrame, methods: list[str]) -> go.Figure:
    fig = go.Figure()
    for m in methods:
        if m not in dd.columns:
            continue
        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd[m] * 100.0,
            mode="lines",
            name=method_label(m),
            line=dict(color=method_color(m), width=1.8),
            fill="tozeroy",
            fillcolor="rgba(0,0,0,0)",
            hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        title="Drawdown paths",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="ρ"),
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Return correlation (daily, full sample)",
        template="plotly_white",
        height=520,
        xaxis=dict(tickangle=-45),
        margin=dict(l=80, r=20, t=60, b=80),
    )
    return fig


def metrics_bar(comp: pd.DataFrame, metric: str, methods: list[str]) -> go.Figure:
    sub = comp.loc[[m for m in methods if m in comp.index], metric]
    fig = go.Figure(go.Bar(
        x=[method_label(m) for m in sub.index],
        y=sub.values,
        marker_color=[method_color(m) for m in sub.index],
        text=[f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}" for v in sub.values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} by method",
        template="plotly_white",
        height=380,
        yaxis_title=metric,
        xaxis_title="",
    )
    return fig


def weights_stacked_area(weights: pd.DataFrame, method: str) -> go.Figure:
    if weights.empty:
        return go.Figure()
    # Re-balance dates are sparse; resample weekly (linear) for a smoother area
    rebal = weights.copy()
    rebal = rebal.asfreq("D").ffill()
    # Cap extreme tails (MVO sometimes hits 0 → 0.30 sharply)
    rebal = rebal.clip(lower=0.0, upper=0.6)

    fig = go.Figure()
    for col in rebal.columns:
        fig.add_trace(go.Scatter(
            x=rebal.index,
            y=rebal[col] * 100.0,
            mode="lines",
            stackgroup="one",
            name=col,
            line=dict(width=0.5, color="white"),
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>w = %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        title=f"Weight history — {method_label(method)}",
        xaxis_title="Date",
        yaxis_title="Portfolio weight (%)",
        template="plotly_white",
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=120),
    )
    return fig


def turnover_chart(turnover: pd.Series, method: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=turnover.index,
        y=turnover.values,
        marker_color=method_color(method),
        hovertemplate="%{x|%Y-%m-%d}<br>Turnover: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Turnover per rebalance — {method_label(method)}",
        xaxis_title="Rebalance date",
        yaxis_title="Turnover (sum |Δw|)",
        template="plotly_white",
        height=320,
    )
    return fig


def cost_sensitivity_chart(cost_df: pd.DataFrame, methods: list[str]) -> go.Figure:
    fig = go.Figure()
    for m in methods:
        if m not in cost_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=cost_df.index,
            y=cost_df[m],
            mode="lines+markers",
            name=method_label(m),
            line=dict(color=method_color(m), width=2.2),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{method_label(m)}</b><br>"
                "Cost: %{x} bps RT<br>"
                "Sharpe: %{y:.3f}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title="Sharpe vs round-trip cost (no risk overlay)",
        xaxis_title="Round-trip cost (bps)",
        yaxis_title="Sharpe ratio",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def robustness_heatmap(sweep: pd.DataFrame) -> go.Figure:
    """Sharpe per (frequency, lookback, method) as a small-multiples heatmap."""
    if sweep.empty:
        return go.Figure()
    freqs = sorted(sweep["frequency"].unique())
    looks = sorted(sweep["lookback_days"].unique())
    methods = [m for m in dl.METHOD_ORDER if m in sweep["method"].unique()]

    fig = go.Figure()
    for m in methods:
        sub = sweep[sweep["method"] == m]
        z = (
            sub.pivot(index="frequency", columns="lookback_days", values="sharpe")
            .reindex(index=freqs, columns=looks)
        )
        fig.add_trace(go.Heatmap(
            z=z.values,
            x=[str(c) for c in z.columns],
            y=z.index,
            zmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="Sharpe"),
            name=method_label(m),
            visible=(m == methods[0]),
            hovertemplate=(
                f"<b>{method_label(m)}</b><br>"
                "freq: %{y}<br>"
                "lookback: %{x} d<br>"
                "Sharpe: %{z:.3f}<extra></extra>"
            ),
        ))

    # Add a dropdown to switch between methods
    buttons = []
    for i, m in enumerate(methods):
        vis = [False] * len(methods)
        vis[i] = True
        buttons.append(dict(
            label=method_label(m),
            method="update",
            args=[{"visible": vis}, {"title": f"Sharpe by (freq × lookback) — {method_label(m)}"}],
        ))
    fig.update_layout(
        template="plotly_white",
        height=340,
        xaxis_title="Lookback (days)",
        yaxis_title="Frequency",
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            showactive=True,
            x=1.15,
            y=1.1,
            xanchor="left",
            yanchor="top",
            buttons=buttons,
        )],
        title=f"Sharpe by (freq × lookback) — {method_label(methods[0])}",
        margin=dict(l=60, r=180, t=70, b=50),
    )
    return fig


def significance_chart(sig: pd.DataFrame) -> go.Figure:
    """Forest plot: point estimate with 95% bootstrap CI per method."""
    if sig.empty:
        return go.Figure()
    methods = [m for m in dl.METHOD_ORDER if m in sig.index]
    sig = sig.loc[methods]
    fig = go.Figure()
    for m in methods:
        row = sig.loc[m]
        color = method_color(m)
        fig.add_trace(go.Scatter(
            x=[row["ci_95_low"], row["ci_95_high"]],
            y=[method_label(m), method_label(m)],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hovertemplate=(
                f"<b>{method_label(m)}</b><br>"
                f"95% CI: %{{x:.3f}}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[row["sharpe"]],
            y=[method_label(m)],
            mode="markers+text",
            marker=dict(color=color, size=11, symbol="diamond"),
            text=[f"  {row['sharpe']:.2f}"],
            textposition="middle right",
            showlegend=False,
            hovertemplate=(
                f"<b>{method_label(m)}</b><br>"
                f"Sharpe: {row['sharpe']:.3f}<br>"
                f"CI: [{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]<extra></extra>"
            ),
        ))
    fig.add_vline(x=0, line=dict(color="grey", width=1, dash="dash"))
    fig.update_layout(
        title="Sharpe with 95% circular-block-bootstrap CI (deflated-SR threshold not crossed)",
        xaxis_title="Sharpe ratio",
        template="plotly_white",
        height=360,
        margin=dict(l=160, r=40, t=60, b=40),
    )
    return fig


def risk_contrib_bar(verification: dict) -> go.Figure:
    """Bar chart of risk contribution shares vs equal target (Risk Parity check)."""
    if not verification:
        return go.Figure()
    tickers = verification.get("tickers", [])
    rc = np.asarray(verification.get("rc_share", []))
    target = np.asarray(verification.get("target_share", []))
    if len(tickers) == 0 or len(rc) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers, y=rc * 100.0, name="Realized",
        marker_color="#1f77b4",
        text=[f"{v:.1f}%" for v in rc * 100.0],
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=tickers, y=target * 100.0, name="Equal target",
        mode="markers",
        marker=dict(symbol="diamond", size=10, color="red"),
    ))
    fig.update_layout(
        title=(
            f"Risk contribution — {verification.get('date','')} | "
            f"max dev = {verification.get('max_deviation', 0):.3f} | "
            f"equal within tol: {verification.get('is_equal_within_tol', False)}"
        ),
        yaxis_title="Risk contribution share (%)",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def stress_bar(stress: pd.DataFrame) -> go.Figure:
    if stress.empty:
        return go.Figure()
    methods = [m for m in dl.METHOD_ORDER if m in stress.index]
    sharpes = stress.loc[methods, "sharpe"].values
    fig = go.Figure(go.Bar(
        x=[method_label(m) for m in methods],
        y=sharpes,
        marker_color=[method_color(m) for m in methods],
        text=[f"{v:.2f}" for v in sharpes],
        textposition="outside",
    ))
    fig.update_layout(
        title="2022 stress test — annualized Sharpe",
        template="plotly_white",
        height=320,
        yaxis_title="Sharpe (2022 only)",
    )
    return fig
