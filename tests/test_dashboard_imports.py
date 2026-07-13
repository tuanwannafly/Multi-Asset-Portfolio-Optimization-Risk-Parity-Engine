"""Import + API-surface smoke test for the dashboard.

Run directly::

    py tests/test_dashboard_imports.py

or as part of the suite::

    py -m pytest tests/test_dashboard_imports.py -v

Stubs out `streamlit` and `plotly` so the test runs in environments
where one or both aren't installed.
"""
import sys
import types
from pathlib import Path

import pytest

# --- stub streamlit ----------------------------------------------------
_fake_st = types.ModuleType("streamlit")
def _noop(*a, **kw): pass
def _decorator(fn=None, **_kw):
    if fn is None: return _decorator
    return fn
class _CtxMgr:
    def __enter__(self): return self
    def __exit__(self, *a): return False
_fake_st.cache_data = _decorator
_fake_st.stop = _noop
_fake_st.warning = _noop
_fake_st.error = _noop
_fake_st.info = _noop
_fake_st.metric = _noop
_fake_st.line_chart = _noop
_fake_st.dataframe = _noop
_fake_st.plotly_chart = _noop
_fake_st.set_page_config = _noop
_fake_st.subheader = _noop
_fake_st.header = _noop
_fake_st.title = _noop
_fake_st.caption = _noop
_fake_st.markdown = _noop
_fake_st.json = _noop
_fake_st.columns = lambda n: [_CtxMgr()] * (n if isinstance(n, int) else len(n))
_fake_st.expander = lambda *a, **kw: _CtxMgr()
_fake_st.multiselect = lambda *a, **kw: kw.get("default", kw.get("options", []))
_fake_st.selectbox = lambda *a, **kw: (kw.get("options") or [None])[0]
_fake_st.slider = lambda *a, **kw: kw.get("value", (None, None))
_fake_st.tabs = lambda labels: [_CtxMgr() for _ in labels]
_fake_st.sidebar = types.SimpleNamespace(
    title=_noop, caption=_noop, error=_noop, warning=_noop,
    multiselect=_fake_st.multiselect, slider=_fake_st.slider,
    expander=lambda *a, **kw: _CtxMgr(),
)
sys.modules.setdefault("streamlit", _fake_st)

# --- stub plotly --------------------------------------------------------
_fake_go = types.ModuleType("plotly.graph_objects")
class _Fig:
    def __init__(self, *a, **kw):
        self.data = []
        self.layout = types.SimpleNamespace(
            title=None, xaxis=None, yaxis=None,
            update_layout=lambda **kw: None,
            add_vline=lambda **kw: None,
        )
    def add_trace(self, trace):
        self.data.append(trace)
        return self
    def update_layout(self, **kw):
        return self
class _Scatter:
    def __init__(self, *a, **kw): pass
class _Bar:
    def __init__(self, *a, **kw): pass
class _Heatmap:
    def __init__(self, *a, **kw): pass
_fake_go.Figure = _Fig
_fake_go.Scatter = _Scatter
_fake_go.Bar = _Bar
_fake_go.Heatmap = _Heatmap
# Patch Figure.add_vline etc. on the class so charts that use them work
# under the stub.
_Fig.add_vline = lambda self, **kw: None
_Fig.add_hline = lambda self, **kw: None
_Fig.add_annotation = lambda self, **kw: None
sys.modules.setdefault("plotly", types.ModuleType("plotly"))
sys.modules.setdefault("plotly.graph_objects", _fake_go)

ROOT = Path("c:/work/New folder (3)").resolve()
sys.path.insert(0, str(ROOT))

import app  # noqa: F401  -- ensures the entrypoint imports cleanly
from dashboard import data_loader, charts  # noqa: F401
from dashboard.tabs import overview, comparison, weights, robustness  # noqa: F401


# ---------------------------------------------------------------------------
# pytest test functions
# ---------------------------------------------------------------------------

def test_app_imports():
    assert app is not None
    assert callable(getattr(app, "main", None))


def test_dashboard_imports():
    import dashboard
    assert dashboard is not None


def test_data_loader_api_surface():
    for name in [
        "load_equity_curves", "load_portfolio_returns", "load_comparison_table",
        "load_weights", "load_diagnostics", "load_robustness_sweep",
        "load_robustness_summary", "load_cost_sensitivity",
        "load_sharpe_significance", "load_stress_2022", "load_efficient_frontier",
        "load_turnover_summary", "load_provenance", "load_risk_parity_verification",
        "drawdown_series", "available_methods", "date_window", "filter_by_window",
        "assemble_corr_from_returns", "METHOD_LABELS", "METHOD_ORDER",
    ]:
        assert hasattr(data_loader, name), f"data_loader.{name} missing"


def test_charts_api_surface():
    for name in [
        "equity_chart", "drawdown_chart", "correlation_heatmap",
        "metrics_bar", "weights_stacked_area", "turnover_chart",
        "cost_sensitivity_chart", "robustness_heatmap", "significance_chart",
        "risk_contrib_bar", "stress_bar",
        "method_color", "method_label", "fmt_pct", "fmt_x", "render_kpi",
    ]:
        assert hasattr(charts, name), f"charts.{name} missing"


@pytest.mark.parametrize("mod,fn", [
    (overview, "render"),
    (comparison, "render"),
    (weights, "render"),
    (robustness, "render"),
])
def test_tab_render_exposed(mod, fn):
    assert callable(getattr(mod, fn, None)), f"{mod.__name__}.{fn} missing or not callable"


def test_method_labels_complete():
    for m in data_loader.METHOD_ORDER:
        assert m in data_loader.METHOD_LABELS


if __name__ == "__main__":
    print("ALL IMPORTS + ATTRIBUTE CHECKS PASS")
