"""Method Comparison tab — side-by-side metrics, risk/return scatter, turnover."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import data_loader as dl
from .. import charts as ch


METRIC_COLS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "effective_n_avg",
    "mean_weight_change",
    "diversification_ratio",
]


def render(methods: list[str]) -> None:
    st.header("Method comparison")
    if not methods:
        st.info("Select at least one method in the sidebar.")
        return
    comp = dl.load_comparison_table()
    if comp.empty:
        st.warning("No `comparison_table.csv` in `results/`. Run `python run.py` first.")
        return

    sub = comp.loc[[m for m in methods if m in comp.index]].copy()
    sub.insert(0, "Method", [ch.method_label(m) for m in sub.index])

    st.subheader("Metric table")
    st.dataframe(
        sub.style.format({
            "annualized_return": "{:.2%}",
            "annualized_volatility": "{:.2%}",
            "max_drawdown": "{:.2%}",
            "sharpe": "{:.3f}",
            "sortino": "{:.3f}",
            "calmar": "{:.3f}",
            "effective_n_avg": "{:.2f}",
            "mean_weight_change": "{:.4f}",
            "diversification_ratio": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ---- Bar charts of single metrics ---------------------------------
    st.subheader("Compare one metric across methods")
    metric = st.selectbox(
        "Metric",
        options=METRIC_COLS,
        index=METRIC_COLS.index("sharpe"),
        format_func=lambda x: x.replace("_", " ").title(),
    )
    st.plotly_chart(ch.metrics_bar(comp, metric, methods), use_container_width=True)

    # ---- Risk / return scatter ----------------------------------------
    st.subheader("Risk vs return")
    if {"annualized_return", "annualized_volatility"}.issubset(comp.columns):
        sel = [m for m in methods if m in comp.index]
        fig = go.Figure()
        for m in sel:
            row = comp.loc[m]
            fig.add_trace(go.Scatter(
                x=[row["annualized_volatility"] * 100.0],
                y=[row["annualized_return"] * 100.0],
                mode="markers+text",
                marker=dict(size=14, color=ch.method_color(m)),
                text=[ch.method_label(m).split(" (")[0]],
                textposition="top center",
                name=ch.method_label(m),
                hovertemplate=(
                    f"<b>{ch.method_label(m)}</b><br>"
                    f"Ann. vol: {row['annualized_volatility']:.2%}<br>"
                    f"Ann. return: {row['annualized_return']:.2%}<br>"
                    f"Sharpe: {row['sharpe']:.3f}<extra></extra>"
                ),
            ))
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Annualized volatility (%)",
            yaxis_title="Annualized return (%)",
            height=420,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Top-left is best (low vol, high return). Risk Parity and Equal "
            "Weight cluster in the low-vol corner; MVO reaches for higher "
            "return at the cost of higher vol."
        )

    # ---- Turnover summary ---------------------------------------------
    st.subheader("Turnover & trading cost")
    turn = dl.load_turnover_summary()
    if turn:
        rows = []
        for m in methods:
            if m not in turn:
                continue
            t = turn[m]
            rows.append({
                "Method": ch.method_label(m),
                "Total turnover": round(t.get("total_turnover", 0.0), 3),
                "Mean turnover / rebalance": round(t.get("mean_turnover_per_rebalance", 0.0), 4),
                "Total cost (decimal)": round(t.get("total_cost", 0.0), 5),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "Total turnover is the sum of |Δw| across every rebalance. "
                "Equal Weight's ≈1 means it traded ~once over the 8-year "
                "window. MVO at ~11.6 traded roughly once per rebalance "
                "month — and HRP at ~9.6 is similar — which is why they get "
                "crushed by transaction costs."
            )
