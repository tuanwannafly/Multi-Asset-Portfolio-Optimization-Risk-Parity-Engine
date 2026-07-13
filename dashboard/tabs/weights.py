"""Weights & Allocation tab — stacked area + turnover + risk-contribution check."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import data_loader as dl
from .. import charts as ch


def render(methods: list[str]) -> None:
    st.header("Weights & allocation")
    st.caption(
        "How each method actually positions the portfolio over time. The "
        "stacked area below fills the full period; turnover bars show one "
        "observation per rebalance date."
    )

    if not methods:
        st.info("Select at least one method in the sidebar.")
        return

    method = st.selectbox(
        "Method to inspect",
        options=methods,
        index=min(1, len(methods) - 1),  # default to second (or first)
        format_func=ch.method_label,
    )

    weights = dl.load_weights(method)
    if weights.empty:
        st.warning(f"No weight history for `{method}`. Did `python run.py` write `weights_{method}.csv`?")
        return

    # ---- Stacked area -------------------------------------------------
    st.subheader(f"Weight history — {ch.method_label(method)}")
    st.plotly_chart(ch.weights_stacked_area(weights, method), use_container_width=True)
    st.caption(
        "Sparse rebalance dates are linearly interpolated between observations "
        "so the area reads as a continuous policy. Individual weight values "
        "shown are at rebalance dates only — see the turnover chart below for "
        "the raw series."
    )

    # ---- Turnover ----------------------------------------------------
    st.subheader("Turnover per rebalance")
    diag = dl.load_diagnostics(method)
    if not diag.empty and "turnover" in diag.columns:
        st.plotly_chart(ch.turnover_chart(diag["turnover"], method), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total turnover", f"{diag['turnover'].sum():.2f}")
        with c2:
            st.metric("Mean turnover / rebalance", f"{diag['turnover'].mean():.3f}")
        with c3:
            st.metric("Max single-rebalance turnover", f"{diag['turnover'].max():.3f}")
    else:
        st.info("No per-rebalance diagnostics found for this method.")

    # ---- Exposure & drawdown path ------------------------------------
    if not diag.empty and {"exposure_scalar", "drawdown"}.issubset(diag.columns):
        st.subheader("Risk overlay state (vol-targeting + drawdown ramp)")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(diag["exposure_scalar"], height=260)
            st.caption("Exposure scalar (1.0 = full risk, 0.0 = fully de-risked).")
        with c2:
            st.line_chart(diag["drawdown"], height=260)
            st.caption("Drawdown path of the equity curve.")

    # ---- Risk-contribution verification ------------------------------
    if method == "risk_parity":
        st.subheader("Risk-contribution verification")
        verif = dl.load_risk_parity_verification()
        if verif:
            st.plotly_chart(ch.risk_contrib_bar(verif), use_container_width=True)
            st.caption(
                "For a true ERC portfolio, every bar should equal 1/N ≈ "
                f"{100.0 / max(1, len(verif.get('tickers', []))):.1f}% (red diamonds). "
                "Max deviation < 0.05 means the Spinu (2013) solver "
                "converged to a numerically equal-risk solution."
            )
        else:
            st.info("No `risk_parity_verification.json` found.")

    # ---- Raw weights table -------------------------------------------
    with st.expander("Raw weight table (per rebalance)"):
        st.dataframe(weights.style.format("{:.4f}"), use_container_width=True)
