"""Robustness tab — sweep heatmap, cost sensitivity, Sharpe significance."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import data_loader as dl
from .. import charts as ch


def render(methods: list[str]) -> None:
    st.header("Robustness & statistical significance")
    st.caption(
        "How much of the headline ranking survives config changes and what "
        "the bootstrap CI actually says about the headline Sharpe."
    )
    if not methods:
        st.info("Select at least one method in the sidebar.")
        return

    # ---- Robustness summary table -----------------------------------
    st.subheader("Stability across (rebalance × lookback) configurations")
    summary = dl.load_robustness_summary()
    if not summary.empty:
        st.dataframe(
            summary.style.format({
                "mean_sharpe": "{:.3f}",
                "std_sharpe": "{:.3f}",
                "min_sharpe": "{:.3f}",
                "max_sharpe": "{:.3f}",
                "range_sharpe": "{:.3f}",
                "n_configs": "{:d}",
            }),
            use_container_width=True,
        )
        st.caption(
            "Each row is the distribution of Sharpe across a 2×3 grid "
            "(monthly/quarterly × 126/252/504-day lookback). Small range ⇒ "
            "the method's ranking is config-insensitive."
        )
    else:
        st.info("No `robustness_summary.csv` found.")

    # ---- Heatmap -----------------------------------------------------
    sweep = dl.load_robustness_sweep()
    if not sweep.empty:
        st.subheader("Sharpe by (rebalance frequency × lookback window)")
        st.plotly_chart(ch.robustness_heatmap(sweep), use_container_width=True)

    # ---- Cost sensitivity -------------------------------------------
    st.subheader("Cost sensitivity (Sharpe vs round-trip cost, no risk overlay)")
    cost = dl.load_cost_sensitivity()
    if not cost.empty:
        st.plotly_chart(ch.cost_sensitivity_chart(cost, methods), use_container_width=True)
        st.caption(
            "With the risk overlay OFF, you can see the raw effect of "
            "transaction costs. MVO and HRP collapse fastest because their "
            "high turnover amplifies the cost drag; Equal Weight is "
            "essentially flat because it trades ~once over the whole window."
        )

        # Side-by-side table
        with st.expander("Cost sensitivity table"):
            st.dataframe(
                cost.style.format("{:.3f}"),
                use_container_width=True,
            )

    # ---- Sharpe significance ----------------------------------------
    st.subheader("Headline Sharpe — bootstrap 95% CI")
    sig = dl.load_sharpe_significance()
    if not sig.empty:
        st.plotly_chart(ch.significance_chart(sig), use_container_width=True)
        st.caption(
            "Diamond = point estimate. Horizontal bar = circular block-bootstrap "
            "95% CI (block length ≈ T^(1/3)). Dashed line at 0 marks the "
            "'is this Sharpe real?' threshold. If the CI straddles 0, you "
            "cannot reject the null that Sharpe = noise on a single window."
        )
        with st.expander("Significance details (t-stat, deflated-SR threshold, ...)"):
            st.dataframe(
                sig.style.format({
                    "sharpe": "{:.3f}",
                    "t_stat_iid": "{:.2f}",
                    "p_value_iid": "{:.4f}",
                    "bootstrap_se": "{:.3f}",
                    "ci_95_low": "{:.3f}",
                    "ci_95_high": "{:.3f}",
                    "block_size": "{:d}",
                    "n_boot": "{:d}",
                    "deflated_sharpe_threshold": "{:.3f}",
                    "prob_sharpe_above_deflated": "{:.2f}",
                    "n_obs": "{:d}",
                }),
                use_container_width=True,
            )
    else:
        st.info("No `sharpe_significance.csv` found.")
