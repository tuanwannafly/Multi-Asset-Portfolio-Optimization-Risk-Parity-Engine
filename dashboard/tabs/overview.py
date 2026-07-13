"""Overview tab — KPIs, equity curves, drawdowns, stress period, provenance."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import data_loader as dl
from .. import charts as ch


def render(equity: pd.DataFrame, returns: pd.DataFrame, methods: list[str]) -> None:
    st.header("Overview")
    st.caption(
        "Walk-forward backtest across 5 allocation methods on a 12-asset "
        "VN30 + GLD + TLT universe. KPIs and charts respect the global date "
        "range and method selection in the sidebar."
    )
    if not methods:
        st.info("Select at least one method in the sidebar.")
        return

    comp = dl.load_comparison_table()
    stress = dl.load_stress_2022()
    dd = dl.drawdown_series(returns)

    # ---- KPIs -----------------------------------------------------------
    st.subheader("Headline metrics")
    cols = st.columns(min(5, len(methods)))
    for col, m in zip(cols, methods):
        if m not in comp.index:
            continue
        row = comp.loc[m]
        with col:
            st.metric(
                label=ch.method_label(m),
                value=ch.fmt_x(row.get("sharpe")),
                delta=f"ret {ch.fmt_pct(row.get('annualized_return'))} • vol {ch.fmt_pct(row.get('annualized_volatility'))}",
                delta_color="off",
                help=f"Max DD: {ch.fmt_pct(row.get('max_drawdown'))}  |  "
                     f"Sortino: {ch.fmt_x(row.get('sortino'))}  |  "
                     f"Calmar: {ch.fmt_x(row.get('calmar'))}",
            )

    # ---- Equity + drawdown side by side --------------------------------
    st.subheader("Equity & drawdown")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(ch.equity_chart(equity, methods), use_container_width=True)
    with c2:
        st.plotly_chart(ch.drawdown_chart(dd, methods), use_container_width=True)

    # ---- 2022 stress ---------------------------------------------------
    st.subheader("2022 stress test")
    if not stress.empty:
        st.plotly_chart(ch.stress_bar(stress), use_container_width=True)
        st.caption(
            "2022 was a high-vol regime for VN equities and US Treasuries. "
            "Risk Parity's loss-of-vol-targeting effect shows up as a deep "
            "negative Sharpe even though it preserves the smoothest drawdown "
            "profile."
        )
    else:
        st.info("2022 stress slice not available for the selected window.")

    # ---- Correlation sanity check --------------------------------------
    st.subheader("Cross-asset correlation")
    corr = dl.assemble_corr_from_returns(returns)
    if not corr.empty:
        st.plotly_chart(ch.correlation_heatmap(corr), use_container_width=True)
        st.caption(
            "VN equities cluster tightly (ρ ≈ 0.5–0.7 between banks), GLD/TLT "
            "are the two most diversifying assets. This is why risk-based "
            "methods (which lean on the correlation structure) behave "
            "materially differently from MVO on this universe."
        )

    # ---- Provenance footer ---------------------------------------------
    with st.expander("Run provenance & data caveats"):
        prov = dl.load_provenance()
        if not prov:
            st.write("No `provenance.json` found in `results/`.")
        else:
            st.json(prov)
