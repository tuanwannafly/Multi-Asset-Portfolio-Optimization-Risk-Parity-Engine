"""Streamlit entrypoint — interactive backtest dashboard.

Designed to run on Hugging Face Spaces (Streamlit runtime) or locally with
``streamlit run app.py``. Reads pre-computed artefacts from ``results/``
produced by ``run.py``.

Run locally::

    streamlit run app.py

The dashboard never re-runs the backtest itself — every chart is a thin
view over the CSVs in ``results/``. To refresh the numbers, re-execute
``python run.py`` (or ``python run.py --real-data``) and reload the page.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable so `from dashboard...` works whether
# Streamlit launches us with CWD=repo or CWD=dashboard/.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard import data_loader as dl
from dashboard import charts as ch
from dashboard.tabs import overview, comparison, weights, robustness


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Portfolio Backtest Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar — global filters
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    st.sidebar.title("📈 Backtest Explorer")
    st.sidebar.caption(
        "Interactive view over the walk-forward backtest results in "
        "`results/`. Re-run `python run.py` to refresh the numbers."
    )

    available = dl.available_methods()
    if not available:
        st.sidebar.error(
            "No `results/comparison_table.csv` found. "
            "Run `python run.py` first."
        )
        st.stop()

    selected = st.sidebar.multiselect(
        "Methods to compare",
        options=available,
        default=available,
        format_func=ch.method_label,
        help="Uncheck a method to remove it from every chart on the page.",
    )
    if not selected:
        st.sidebar.warning("Select at least one method.")
        st.stop()

    # Date window — use the equity curves as the source of truth
    equity = dl.load_equity_curves()
    if not equity.empty:
        min_date = equity.index.min().to_pydatetime()
        max_date = equity.index.max().to_pydatetime()
        dates = st.sidebar.slider(
            "Date range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
        )
    else:
        dates = None

    with st.sidebar.expander("About this dashboard"):
        st.markdown(
            "**Underlying engine:** `run.py` — walk-forward portfolio "
            "backtest across 4 allocation methods (MVO, Risk Parity, HRP, "
            "Black-Litterman) plus an Equal-Weight baseline.\n\n"
            "**Data:** VN30 large-caps (vnstock) + GLD/TLT (yfinance HTTP).\n\n"
            "**Source:** see `README.md` and `RESEARCH_NOTE.md` in the repo."
        )
        st.markdown(
            "[`README.md`](README.md) · "
            "[`RESEARCH_NOTE.md`](RESEARCH_NOTE.md)"
        )

    return {"methods": selected, "dates": dates}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    state = render_sidebar()
    methods = state["methods"]
    dates = state["dates"]

    # Load + filter the two time-series once and pass them into tabs
    equity = dl.load_equity_curves()
    returns = dl.load_portfolio_returns()
    if dates is not None:
        start, end = dates
        equity = dl.filter_by_window(equity, start, end)
        returns = dl.filter_by_window(returns, start, end)

    if equity.empty or returns.empty:
        st.error("Selected date window is empty. Widen the slider in the sidebar.")
        st.stop()

    # Title + subtitle
    st.title("Portfolio backtest explorer")
    st.caption(
        f"Window: **{equity.index.min().date()} → {equity.index.max().date()}** "
        f"({len(equity):,} trading days) · "
        f"Methods: **{', '.join(ch.method_label(m) for m in methods)}**"
    )

    tabs = st.tabs(["Overview", "Method comparison", "Weights & allocation", "Robustness"])

    with tabs[0]:
        overview.render(equity, returns, methods)
    with tabs[1]:
        comparison.render(methods)
    with tabs[2]:
        weights.render(methods)
    with tabs[3]:
        robustness.render(methods)


# Streamlit executes the script top-to-bottom on every rerun, so main()
# is called unconditionally. The `if __name__ == "__main__"` guard is
# kept so the file can be imported for tests without re-running the app.
main()
