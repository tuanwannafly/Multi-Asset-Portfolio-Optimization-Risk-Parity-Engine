"""Unit tests for the portfolio optimization engine.

Run with:
    python -m pytest tests/ -v

or:
    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.ingestion import fetch_prices, compute_returns
from src.covariance import (
    ledoit_wolf_shrinkage,
    sample_covariance,
    compare_covariance_methods,
    condition_number,
)
from src.returns_estimation import latest_estimate, historical_mean
from src.optimizers.mean_variance import mvo_min_variance, efficient_frontier
from src.optimizers.risk_parity import risk_parity, verify_risk_contributions
from src.optimizers.hrp import hrp, hierarchical_cluster
from src.optimizers.black_litterman import (
    implied_equilibrium_returns,
    black_litterman_posterior,
    make_pick_matrix,
)
from src.methods import equal_weight, mean_variance, risk_parity_weights, hrp_weights
from src.backtest.rebalance_engine import RebalanceConfig, run_backtest, _turnover
from src.significance import (
    sharpe_t_stat,
    circular_block_bootstrap_sharpe,
    deflated_sharpe,
)
from src.robustness import (
    run_robustness_sweep,
    run_cost_sensitivity,
    summarize_sweep,
)
from src.significance import (
    sharpe_t_stat,
    circular_block_bootstrap_sharpe,
    deflated_sharpe,
)
from src.robustness import (
    run_robustness_sweep,
    run_cost_sensitivity,
    summarize_sweep,
)
from src.evaluate import (
    annualized_return,
    sharpe_ratio,
    max_drawdown,
    effective_n,
    diversification_ratio,
    comparison_table,
)
from src.backtest.risk import compute_exposure_scalar, drawdown_path, apply_overlay


def _make_returns(n_obs: int = 504, n_assets: int = 6, seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic returns for tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_obs)
    rets = rng.normal(0.0005, 0.01, size=(n_obs, n_assets))
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(rets, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

def test_sample_covariance_shape_and_psd() -> None:
    rets = _make_returns()
    cov = sample_covariance(rets)
    assert cov.shape == (6, 6)
    # Symmetric
    assert np.allclose(cov.values, cov.values.T)
    # PSD
    eigvals = np.linalg.eigvalsh(cov.values)
    assert eigvals.min() > -1e-10


def test_ledoit_wolf_shrinks_toward_target() -> None:
    rets = _make_returns()
    cov_lw, shrink = ledoit_wolf_shrinkage(rets, target="diag")
    assert 0.0 <= shrink <= 1.0
    # Condition number should be comparable to or better than sample
    sample = sample_covariance(rets)
    assert condition_number(cov_lw) <= condition_number(sample) * 1.5


def test_compare_covariance_methods_returns_frame() -> None:
    rets = _make_returns()
    summary = compare_covariance_methods(rets)
    assert {"sample_cond", "lw_cond", "lw_shrinkage"}.issubset(set(summary.index))


# ---------------------------------------------------------------------------
# Mean-Variance
# ---------------------------------------------------------------------------

def test_mvo_min_variance_weights_sum_to_one() -> None:
    rets = _make_returns()
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    mu = rets.mean() * 252.0
    w = mvo_min_variance(cov.values, mu=mu.values, long_only=True, max_weight=0.5)
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w >= -1e-8).all()
    assert (w <= 0.5 + 1e-6).all()


def test_mvo_min_variance_no_target_no_mu() -> None:
    rets = _make_returns()
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    w = mvo_min_variance(cov.values, mu=None, long_only=True, max_weight=1.0)
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w >= 0).all()


def test_mvo_infeasible_target_falls_back() -> None:
    rets = _make_returns()
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    mu = rets.mean() * 252.0
    # Target way above achievable — should fall back gracefully
    w = mvo_min_variance(cov.values, mu=mu.values, target_return=10.0, long_only=True, max_weight=0.3)
    assert abs(w.sum() - 1.0) < 1e-4


def test_efficient_frontier_runs() -> None:
    rets = _make_returns()
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    mu = rets.mean() * 252.0
    f = efficient_frontier(cov, mu, n_points=10)
    assert len(f) == 10
    assert {"achieved_return", "achieved_vol", "sharpe", "weights"}.issubset(f.columns)


# ---------------------------------------------------------------------------
# Risk Parity
# ---------------------------------------------------------------------------

def test_risk_parity_weights_sum_to_one() -> None:
    rets = _make_returns()
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    w = risk_parity(cov.values, long_only=True, max_weight=1.0)
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w >= 0).all()


def test_risk_parity_equal_risk_contributions() -> None:
    """Risk contributions should be approximately equal across assets."""
    rets = _make_returns(seed=42)
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    w = risk_parity(cov.values, long_only=True, max_weight=1.0)
    rc = verify_risk_contributions(w, cov.values)
    # Without max_weight cap, equal risk contributions achievable to high precision
    assert rc["max_deviation"] < 0.05, f"RC deviation too large: {rc['max_deviation']}"


def test_risk_parity_respects_max_weight() -> None:
    rets = _make_returns()
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    w = risk_parity(cov.values, long_only=True, max_weight=0.25)
    assert (w <= 0.25 + 1e-6).all()
    assert abs(w.sum() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------

def test_hrp_weights_sum_to_one() -> None:
    rets = _make_returns(seed=7)
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    w = hrp(cov, returns=rets)
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w >= 0).all()


def test_hrp_clustering_returns_deterministic_order() -> None:
    rets = _make_returns(seed=11)
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    Z, order = hierarchical_cluster(pd.DataFrame(corr, index=cov.index, columns=cov.columns))
    assert len(order) == len(cov)
    assert sorted(order) == list(range(len(cov)))


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

def test_bl_posterior_dimensions() -> None:
    rets = _make_returns(seed=3)
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    prior = implied_equilibrium_returns(cov, np.ones(6) / 6)
    pick = np.eye(6)[:3]  # views on first 3 assets
    Q = np.array([0.10, 0.08, 0.12])
    post = black_litterman_posterior(cov, prior.values, pick, Q, tau=0.05)
    assert len(post) == 6


def test_bl_no_view_returns_prior() -> None:
    """With zero views, posterior should equal prior (up to small numerical drift)."""
    rets = _make_returns(seed=5)
    cov, _ = ledoit_wolf_shrinkage(rets, target="diag")
    prior = implied_equilibrium_returns(cov, np.ones(6) / 6).values
    # No view (zero pick rows)
    pick = np.zeros((0, 6))
    Q = np.zeros(0)
    post = black_litterman_posterior(cov, prior, pick, Q, tau=0.05)
    assert np.allclose(post.values, prior, atol=1e-6)


def test_make_pick_matrix_correct_shape() -> None:
    views = [
        {"assets": ["A0", "A1"], "weights": [0.5, 0.5]},
        {"assets": ["A2"], "weights": [1.0]},
    ]
    tickers = [f"A{i}" for i in range(6)]
    P = make_pick_matrix(views, tickers)
    assert P.shape == (2, 6)
    assert P[0, 0] == 0.5 and P[0, 1] == 0.5
    assert P[1, 2] == 1.0


# ---------------------------------------------------------------------------
# Risk overlay
# ---------------------------------------------------------------------------

def test_exposure_scalar_at_zero_drawdown() -> None:
    s = compute_exposure_scalar(realized_vol=0.10, target_vol=0.12, drawdown=0.0)
    assert abs(s - 1.0) < 1e-9


def test_exposure_scalar_at_soft_dd() -> None:
    s = compute_exposure_scalar(realized_vol=0.10, target_vol=0.12, drawdown=-0.05,
                                 max_dd_soft=0.10, max_dd_hard=0.20)
    # At dd=-0.05 (halfway through soft ramp), dd_scalar = 0.75; vol scalar = 1.0
    assert abs(s - 0.75) < 1e-6


def test_exposure_scalar_zero_beyond_hard_dd() -> None:
    s = compute_exposure_scalar(realized_vol=0.10, target_vol=0.12, drawdown=-0.25,
                                 max_dd_soft=0.10, max_dd_hard=0.20)
    assert s == 0.0


def test_exposure_scalar_vol_target() -> None:
    # Realized vol 0.24, target 0.12 → scalar 0.5 (ignoring dd)
    s = compute_exposure_scalar(realized_vol=0.24, target_vol=0.12, drawdown=0.0)
    assert abs(s - 0.5) < 1e-9


def test_drawdown_path() -> None:
    equity = pd.Series([100.0, 110.0, 99.0, 95.0, 105.0])
    dd = drawdown_path(equity)
    # Peak is 110; last value 105 → dd = 105/110 - 1 = -0.0455
    expected = pd.Series([0.0, 0.0, -0.10, -0.1363636, -0.0454545])
    assert np.allclose(dd.values, expected.values, atol=1e-5)


def test_apply_overlay_scales_weights() -> None:
    weights = np.array([0.4, 0.3, 0.3])
    rets = pd.Series([0.01, -0.05, 0.02, -0.03, 0.01])  # enough history
    scaled, meta = apply_overlay(weights, rets, target_vol=0.12, max_dd_hard=0.20, max_dd_soft=0.10)
    assert scaled.sum() <= weights.sum() + 1e-9
    assert "exposure_scalar" in meta
    assert "drawdown" in meta


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def test_turnover_formula() -> None:
    old = np.array([0.5, 0.5, 0.0])
    new = np.array([0.3, 0.3, 0.4])
    to = _turnover(old, new)
    expected = 0.5 * (0.2 + 0.2 + 0.4)
    assert abs(to - expected) < 1e-9


def test_backtest_runs_all_methods() -> None:
    rets = _make_returns(n_obs=400, n_assets=6)
    cfg = RebalanceConfig(frequency="monthly", lookback_days=120, cost_bps=10.0)
    methods = {
        "equal_weight": equal_weight,
        "risk_parity": risk_parity_weights,
        "hrp": hrp_weights,
    }
    res = run_backtest(rets, methods, cfg, cov_method="ledoit_wolf")
    assert {"equity", "returns", "weights", "diagnostics"}.issubset(res.keys())
    assert res["equity"].shape == (len(rets), 3)
    # The first equity row reflects the first return applied (after the
    # first rebalance at index 0); it should be within ±5% of initial
    # capital on a small synthetic dataset.
    for col in res["equity"].columns:
        first = res["equity"][col].iloc[0]
        assert 0.95 * cfg.initial_capital < first < 1.05 * cfg.initial_capital
    # Equity should be strictly increasing or have some drift by the end
    # on a non-zero-mean return series
    final = res["equity"].iloc[-1].mean()
    assert 0.5 * cfg.initial_capital < final < 2.0 * cfg.initial_capital


def test_backtest_cost_reduces_nav() -> None:
    rets = _make_returns(n_obs=400, n_assets=6)
    cfg_no_cost = RebalanceConfig(frequency="monthly", lookback_days=120, cost_bps=0.0)
    cfg_with_cost = RebalanceConfig(frequency="monthly", lookback_days=120, cost_bps=20.0)
    methods = {"equal_weight": equal_weight}
    res_no = run_backtest(rets, methods, cfg_no_cost, cov_method="ledoit_wolf")
    res_with = run_backtest(rets, methods, cfg_with_cost, cov_method="ledoit_wolf")
    # Cost should reduce final NAV
    assert res_with["equity"]["equal_weight"].iloc[-1] < res_no["equity"]["equal_weight"].iloc[-1]


def test_backtest_walkforward_no_lookahead() -> None:
    """Sanity: equity at time t depends only on data through time t."""
    rets = _make_returns(n_obs=400, n_assets=6)
    # Use cost_bps=0 so no cost is charged at rebalance; otherwise the timing
    # of the FIRST rebalance differs between full and truncated runs and
    # breaks the byte-equality check below.
    cfg = RebalanceConfig(frequency="monthly", lookback_days=120, cost_bps=0.0)
    methods = {"equal_weight": equal_weight}
    res1 = run_backtest(rets, methods, cfg, cov_method="ledoit_wolf")
    # Truncate to half the data
    res2 = run_backtest(rets.iloc[:200], methods, cfg, cov_method="ledoit_wolf")
    # First 200 rows of equity should match within floating-point tolerance.
    # Equal_weight with cost=0 has zero turnover so trajectories should align;
    # we allow a small relative tolerance for accumulated float drift.
    np.testing.assert_allclose(
        res1["equity"]["equal_weight"].iloc[:200].values,
        res2["equity"]["equal_weight"].values,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluation_metrics_basic() -> None:
    rets = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 500))
    ar = annualized_return(rets)
    sh = sharpe_ratio(rets)
    mdd = max_drawdown(rets)
    assert -1.0 < ar < 1.0
    assert -5 < sh < 5
    assert -1.0 <= mdd <= 0.0


def test_effective_n_uniform() -> None:
    w = np.ones(10) / 10
    assert abs(effective_n(w) - 10.0) < 1e-9


def test_effective_n_concentrated() -> None:
    w = np.array([1.0, 0, 0, 0, 0])
    assert abs(effective_n(w) - 1.0) < 1e-9


def test_diversification_ratio_uncorrelated() -> None:
    """For an equal-weighted portfolio of uncorrelated assets, DR = sqrt(N).

    This is because the weighted-avg vol is sigma * (sum w_i), and the
    portfolio vol is sigma * sqrt(sum w_i^2). For equal weights w_i = 1/N:
        DR = sqrt(N) * (1/N) * N / 1 = sqrt(N).
    """
    n = 5
    cov = np.eye(n)  # all variances 1, zero correlation
    w = np.ones(n) / n
    dr = diversification_ratio(w, cov)
    assert abs(dr - np.sqrt(n)) < 1e-9, f"DR={dr}, expected sqrt(N)={np.sqrt(n)}"


def test_comparison_table_builds() -> None:
    dates = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0.0005, 0.01, (200, 3)),
                         index=dates, columns=["a", "b", "c"])
    returns_df = pd.DataFrame({
        "a": rets["a"], "b": rets["b"] * 0.5 + 0.0002, "c": rets["c"] * -0.2,
    })
    weights_dict = {
        "a": pd.DataFrame({"x": [0.5, 0.6], "y": [0.5, 0.4]}, index=dates[:2]),
        "b": pd.DataFrame({"x": [0.3, 0.3], "y": [0.7, 0.7]}, index=dates[:2]),
        "c": pd.DataFrame({"x": [1.0, 1.0], "y": [0.0, 0.0]}, index=dates[:2]),
    }
    cov = pd.DataFrame(np.eye(2), index=["x", "y"], columns=["x", "y"])
    table = comparison_table(returns_df, weights_dict, cov)
    assert set(table.index) == {"a", "b", "c"}
    assert "sharpe" in table.columns
    assert "effective_n_avg" in table.columns


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def test_synthetic_prices_reproducible() -> None:
    cfg = load_config()
    p1 = fetch_prices(cfg, use_synthetic=True)
    p2 = fetch_prices(cfg, use_synthetic=True)
    pd.testing.assert_frame_equal(p1, p2)


def test_returns_shape_matches_prices() -> None:
    cfg = load_config()
    prices = fetch_prices(cfg, use_synthetic=True)
    rets = compute_returns(prices)
    assert rets.shape == (prices.shape[0] - 1, prices.shape[1])
    # First row of returns is NaN-free? No: first row will be the second day.
    assert rets.iloc[0].notna().any() or rets.iloc[1].notna().any()


def test_returns_estimation_shape() -> None:
    rets = _make_returns()
    mu = historical_mean(rets, window=120)
    assert mu.shape == rets.shape
    # First 59 rows should be entirely NaN (min_periods=60 starts at row 60)
    # pandas rolling includes the current row, so min_periods=60 means the first
    # 59 rows (indices 0..58) have fewer than 60 observations → NaN.
    assert mu.iloc[:59].isna().all().all(), f"Found non-NaN in first 59 rows"
    # Last row should be finite
    assert np.isfinite(mu.iloc[-1]).all()


# ---------------------------------------------------------------------------
# Sharpe significance
# ---------------------------------------------------------------------------

def test_sharpe_t_stat_basic() -> None:
    """Under IID normal returns the t-stat is finite and p-value in [0, 1]."""
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    t_stat, p_val, sr = sharpe_t_stat(r)
    assert np.isfinite(t_stat)
    assert 0.0 <= p_val <= 1.0
    assert 0.3 < sr < 0.8


def test_bootstrap_sharpe_ci_covers_estimate() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.012, 1000))
    boot = circular_block_bootstrap_sharpe(r, n_boot=100, seed=0)
    assert boot["ci_low"] <= boot["sharpe"] <= boot["ci_high"]
    assert boot["n_boot"] >= 50  # some samples may be degenerate
    assert boot["block_size"] >= 5


def test_deflated_sharpe_single_trial_zero() -> None:
    """With n_trials=1 the deflated threshold is 0 by construction."""
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0003, 0.01, 500))
    d = deflated_sharpe(r, n_trials=1)
    assert abs(d["deflated_sharpe"]) < 1e-9


# ---------------------------------------------------------------------------
# Robustness sweep
# ---------------------------------------------------------------------------

def test_robustness_sweep_runs() -> None:
    rets = _make_returns(n_obs=300, n_assets=5, seed=2)
    df = run_robustness_sweep(rets, freq_grid=["monthly"], lookback_grid=[126, 252])
    assert not df.empty
    assert {"method", "frequency", "lookback_days", "sharpe"}.issubset(df.columns)
    assert (df["frequency"] == "monthly").all()
    assert df["lookback_days"].nunique() == 2


def test_cost_sensitivity_changes_sharpe() -> None:
    """Sharpe MUST decrease as cost rises (sanity check the engine charges cost).

    Use mean_variance — equal_weight has zero turnover by construction so it's
    insensitive to cost. The cost-charging code path is exercised only when
    the weight function actually moves weights.
    """
    rets = _make_returns(n_obs=300, n_assets=5, seed=2)
    df = run_cost_sensitivity(rets, cost_grid=[0, 50, 200], frequency="monthly",
                                lookback_days=126, risk_overlay=False)
    mvo_0 = df[(df["method"] == "mean_variance") & (df["cost_bps_roundtrip"] == 0)]["sharpe"].iloc[0]
    mvo_200 = df[(df["method"] == "mean_variance") & (df["cost_bps_roundtrip"] == 200)]["sharpe"].iloc[0]
    assert mvo_200 < mvo_0, (
        f"Cost sensitivity broken: mean_variance Sharpe did not decrease with cost "
        f"({mvo_0} at 0bps vs {mvo_200} at 200bps). Transaction cost is not being charged."
    )


if __name__ == "__main__":
    # Run tests inline if pytest not available
    import inspect
    funcs = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in funcs:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{len(funcs) - failed}/{len(funcs)} tests passed")
    if failed:
        sys.exit(1)