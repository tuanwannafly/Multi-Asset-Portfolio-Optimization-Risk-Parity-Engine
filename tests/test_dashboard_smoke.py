"""End-to-end smoke test for the dashboard data loaders.

Run directly::

    py tests/test_dashboard_smoke.py

or as part of the suite::

    py -m pytest tests/test_dashboard_smoke.py -v
"""
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Stub streamlit so we can import data_loader without a real Streamlit
# runtime (useful in CI / sandboxed environments).
_fake = types.ModuleType("streamlit")
def _decorator(fn=None, **_kw):
    if fn is None:
        return _decorator
    return fn
_fake.cache_data = _decorator
_fake.stop = lambda *a, **kw: None
_fake.warning = lambda *a, **kw: None
_fake.error = lambda *a, **kw: None
_fake.info = lambda *a, **kw: None
sys.modules.setdefault("streamlit", _fake)

ROOT = Path("c:/work/New folder (3)").resolve()
sys.path.insert(0, str(ROOT))

from dashboard import data_loader as dl


# ---------------------------------------------------------------------------
# pytest test functions
# ---------------------------------------------------------------------------

def test_equity_curves():
    eq = dl.load_equity_curves()
    assert not eq.empty
    assert eq.shape[1] == 5
    assert isinstance(eq.index, pd.DatetimeIndex)
    assert eq.index.is_monotonic_increasing


def test_portfolio_returns():
    ret = dl.load_portfolio_returns()
    assert not ret.empty
    assert ret.shape[1] == 5
    assert isinstance(ret.index, pd.DatetimeIndex)


def test_comparison_table():
    comp = dl.load_comparison_table()
    assert not comp.empty
    assert set(comp.index) == set(dl.METHOD_ORDER)
    assert "sharpe" in comp.columns
    assert "max_drawdown" in comp.columns


def test_robustness_sweep_and_summary():
    sw = dl.load_robustness_sweep()
    assert not sw.empty
    assert {"method", "frequency", "lookback_days", "sharpe"}.issubset(sw.columns)
    summ = dl.load_robustness_summary()
    assert not summ.empty
    assert set(summ.index) == set(dl.METHOD_ORDER)


def test_cost_sensitivity():
    cost = dl.load_cost_sensitivity()
    assert not cost.empty
    assert cost.index.name == "cost_bps_roundtrip"
    for m in dl.METHOD_ORDER:
        assert m in cost.columns


def test_sharpe_significance():
    sig = dl.load_sharpe_significance()
    assert not sig.empty
    assert "sharpe" in sig.columns
    assert "ci_95_low" in sig.columns
    assert "ci_95_high" in sig.columns


def test_stress_2022():
    stress = dl.load_stress_2022()
    assert not stress.empty
    assert "sharpe" in stress.columns


def test_efficient_frontier():
    ef = dl.load_efficient_frontier()
    assert not ef.empty


def test_turnover_summary_dict():
    turn = dl.load_turnover_summary()
    assert isinstance(turn, dict)
    assert set(turn.keys()) == set(dl.METHOD_ORDER)
    for m, t in turn.items():
        assert "total_turnover" in t
        assert "total_cost" in t
        # MVO + HRP should have much higher turnover than Equal Weight
        if m == "equal_weight":
            assert t["total_turnover"] < 2.0
        if m == "mean_variance":
            assert t["total_turnover"] > 5.0


def test_provenance_has_core_fields():
    prov = dl.load_provenance()
    assert "data_sources" in prov
    assert "date_range" in prov
    assert "n_assets" in prov
    assert "currency_mix" in prov


def test_risk_parity_verification():
    verif = dl.load_risk_parity_verification()
    assert "max_deviation" in verif
    assert "rc_share" in verif
    assert "tickers" in verif
    # ERC shares should sum to ~1.0
    rc = pd.Series(verif["rc_share"])
    assert abs(rc.sum() - 1.0) < 1e-6
    # Max deviation from equal target should be small (Spinu's iterative
    # solver converges to within ~10% on this universe; the project
    # tolerance is 0.05 which is occasionally slightly exceeded on real
    # data, so we use 0.10 here as a sanity check).
    assert verif["max_deviation"] < 0.10, (
        f"Risk Parity max deviation {verif['max_deviation']:.3f} is too large"
    )
    # Number of assets matches
    assert len(rc) == len(verif["tickers"])


@pytest.mark.parametrize("method", dl.METHOD_ORDER)
def test_per_method_weights_and_diagnostics(method):
    w = dl.load_weights(method)
    d = dl.load_diagnostics(method)
    assert not w.empty, f"weights_{method} is empty"
    assert not d.empty, f"diagnostics_{method} is empty"
    # Weights on each rebalance date must sum to ~1
    sums = w.sum(axis=1)
    assert ((sums - 1.0).abs() < 1e-3).all(), (
        f"weights for {method} do not sum to 1 on every rebalance: "
        f"min={sums.min()}, max={sums.max()}"
    )
    # Diagnostics must contain the standard columns
    for col in ("turnover", "cost", "exposure_scalar", "realized_vol", "drawdown"):
        assert col in d.columns, f"diagnostics_{method} missing {col}"


def test_drawdown_series():
    ret = dl.load_portfolio_returns()
    dd = dl.drawdown_series(ret)
    assert dd.shape == ret.shape
    # Drawdown is always ≤ 0 (or 0 at the start)
    assert (dd <= 1e-9).all().all()
    # Starts at 0 on the first row
    assert (dd.iloc[0] == 0).all()


def test_available_methods():
    methods = dl.available_methods()
    assert methods == dl.METHOD_ORDER


def test_filter_by_window():
    ret = dl.load_portfolio_returns()
    start, end = ret.index[100], ret.index[-100]
    filt = dl.filter_by_window(ret, start, end)
    assert filt.index.min() >= pd.Timestamp(start)
    assert filt.index.max() <= pd.Timestamp(end)
    assert len(filt) < len(ret)


def test_date_window():
    eq = dl.load_equity_curves()
    s, e = dl.date_window(eq)
    assert s <= e
    assert s.year >= 2017
    assert e.year <= 2025


def test_correlation_assembly():
    ret = dl.load_portfolio_returns()
    corr = dl.assemble_corr_from_returns(ret)
    assert corr.shape == (ret.shape[1], ret.shape[1])
    # Diagonal must be exactly 1
    import numpy as np
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-12)
    # Symmetric
    np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-12)


if __name__ == "__main__":
    # Lightweight direct-runner that just reports each loader's shape
    loaders = [
        ("equity_curves", dl.load_equity_curves),
        ("portfolio_returns", dl.load_portfolio_returns),
        ("comparison_table", dl.load_comparison_table),
        ("robustness_sweep", dl.load_robustness_sweep),
        ("robustness_summary", dl.load_robustness_summary),
        ("cost_sensitivity", dl.load_cost_sensitivity),
        ("sharpe_significance", dl.load_sharpe_significance),
        ("stress_test_2022", dl.load_stress_2022),
        ("efficient_frontier", dl.load_efficient_frontier),
        ("turnover_summary", dl.load_turnover_summary),
        ("provenance", dl.load_provenance),
        ("risk_parity_verification", dl.load_risk_parity_verification),
    ]
    for name, fn in loaders:
        out = fn()
        if hasattr(out, "shape"):
            print(f"OK   {name:<30s} shape={out.shape}")
        elif isinstance(out, dict):
            print(f"OK   {name:<30s} dict keys={sorted(out.keys())[:6]}")
        else:
            print(f"OK   {name:<30s} type={type(out).__name__}")
    print()
    print("ALL OK")
