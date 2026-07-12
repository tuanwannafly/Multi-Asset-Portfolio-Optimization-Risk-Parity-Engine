"""End-to-end runner: ingest → estimate → optimize → backtest → evaluate.

Usage:
    python run.py                # uses synthetic data (reproducible)
    python run.py --real-data    # try real data first, fall back to synthetic
    python run.py --years 3      # override lookback horizon
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from anywhere
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, repo_path
from src.ingestion import fetch_prices, compute_returns, align_calendar, save_processed
from src.covariance import (
    ledoit_wolf_shrinkage,
    sample_covariance,
    compare_covariance_methods,
    condition_number,
)
from src.returns_estimation import historical_mean, latest_estimate
from src.optimizers.mean_variance import efficient_frontier
from src.optimizers.hrp import hierarchical_cluster
from src.optimizers.black_litterman import (
    implied_equilibrium_returns,
    black_litterman_posterior,
    make_pick_matrix,
)
from src.optimizers.risk_parity import verify_risk_contributions
from src.methods import (
    equal_weight,
    mean_variance,
    risk_parity_weights,
    hrp_weights,
    black_litterman_weights_fn,
    METHOD_REGISTRY,
)
from src.backtest.rebalance_engine import RebalanceConfig, run_backtest
from src.evaluate import comparison_table, summarize
from src import visualize
from src import robustness
from src import significance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
LOGGER = logging.getLogger("run")


def main(args: argparse.Namespace) -> None:
    cfg = load_config()
    results_dir = repo_path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 1 — Data ingestion")
    LOGGER.info("=" * 60)
    prices = fetch_prices(cfg, use_synthetic=args.synthetic)
    prices = align_calendar(prices)
    returns = compute_returns(prices, method="log")
    save_processed(prices, returns)
    LOGGER.info("Universe: %d assets, %d trading days", prices.shape[1], len(prices))

    # ------------------------------------------------------------------
    # 2. Covariance comparison (motivation for shrinkage)
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 2 — Covariance comparison (sample vs Ledoit-Wolf)")
    LOGGER.info("=" * 60)
    cov_summary = compare_covariance_methods(returns)
    print("\nCovariance comparison:")
    print(cov_summary.to_string())
    cov_summary.to_csv(results_dir / "covariance_comparison.csv")

    # ------------------------------------------------------------------
    # 3. Correlation & dendrogram
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 3 — Correlation structure & hierarchical clustering")
    LOGGER.info("=" * 60)
    cov_full, _ = ledoit_wolf_shrinkage(returns, target="diag")
    std = np.sqrt(np.diag(cov_full.values))
    corr = cov_full.values / np.outer(std, std)
    corr_df = pd.DataFrame(corr, index=prices.columns, columns=prices.columns)
    visualize.plot_correlation_heatmap(corr_df, results_dir / "correlation_heatmap.png")
    Z, order = hierarchical_cluster(corr_df)
    visualize.plot_dendrogram(Z, list(corr_df.columns), results_dir / "dendrogram.png")

    # ------------------------------------------------------------------
    # 4. Efficient frontier (in-sample)
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 4 — Efficient frontier (in-sample)")
    LOGGER.info("=" * 60)
    mu_full = latest_estimate(returns, window=cfg["returns"]["rolling_window"])
    if mu_full is not None and not mu_full.isna().all():
        mu_vec = mu_full.fillna(mu_full.median()).reindex(prices.columns)
    else:
        # Fallback: use historical mean over full sample
        mu_vec = returns.mean() * 252.0
    frontier = efficient_frontier(cov_full, mu_vec, n_points=30)
    frontier_only = frontier.drop(columns=["weights"])
    frontier_only.to_csv(results_dir / "efficient_frontier.csv")
    visualize.plot_efficient_frontier(frontier, results_dir / "efficient_frontier.png")

    # ------------------------------------------------------------------
    # 5. Walk-forward backtest
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 5 — Walk-forward backtest")
    LOGGER.info("=" * 60)
    rb_cfg = RebalanceConfig(
        frequency=cfg["rebalance"]["frequency"],
        lookback_days=cfg["rebalance"]["lookback_days"],
        cost_bps=cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"],
        long_only=cfg["constraints"]["long_only"],
        max_weight=cfg["constraints"]["max_weight"],
        min_weight=cfg["constraints"]["min_weight"],
        target_return_annual=cfg["mvo"]["target_return_annual"],
        risk_overlay=cfg["risk_overlay"]["enabled"],
        target_vol_annual=cfg["risk_overlay"]["target_vol_annual"],
        vol_lookback_days=cfg["risk_overlay"]["vol_lookback_days"],
        max_dd_soft=cfg["risk_overlay"].get("max_dd_soft", 0.10),
        max_dd_hard=cfg["risk_overlay"]["max_drawdown_kill_switch"],
        initial_capital=cfg["backtest"]["initial_capital"],
    )

    methods: dict = {}
    for m in cfg["methods"]:
        if m == "black_litterman":
            # Construct views from a simple momentum signal (12-1 month) for illustration
            momentum = (returns.rolling(252).mean() - returns.rolling(21).mean()).iloc[-1]
            views = [{"assets": [t], "weights": [1.0]} for t in momentum.index]
            view_returns = (momentum.fillna(0).clip(-0.05, 0.10) * 12).tolist()
            confidences = [0.4] * len(views)
            methods[m] = black_litterman_weights_fn(
                views=views,
                view_returns=view_returns,
                confidences=confidences,
                tau=cfg["black_litterman"]["tau"],
            )
        elif m in METHOD_REGISTRY:
            methods[m] = METHOD_REGISTRY[m]
        else:
            LOGGER.warning("Unknown method '%s' — skipping", m)

    backtest_result = run_backtest(
        returns,
        methods=methods,
        cfg=rb_cfg,
        cov_method=cfg["covariance"]["method"],
    )

    equity = backtest_result["equity"]
    port_returns = backtest_result["returns"]
    weights_dict = backtest_result["weights"]
    diagnostics_dict = backtest_result["diagnostics"]

    equity.to_csv(results_dir / "equity_curves.csv")
    port_returns.to_csv(results_dir / "portfolio_returns.csv")
    for m, df in weights_dict.items():
        df.to_csv(results_dir / f"weights_{m}.csv")
    for m, df in diagnostics_dict.items():
        df.to_csv(results_dir / f"diagnostics_{m}.csv")

    # ------------------------------------------------------------------
    # 6. Evaluation & comparison
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 6 — Evaluation")
    LOGGER.info("=" * 60)
    comp = comparison_table(port_returns, weights_dict, cov_full)
    comp.to_csv(results_dir / "comparison_table.csv")
    print("\n=== Backtest comparison ===")
    print(comp.to_string())

    # Total turnover per method
    turnover_summary = {
        m: {
            "total_turnover": float(df["turnover"].sum()) if not df.empty else 0.0,
            "mean_turnover_per_rebalance": float(df["turnover"].mean()) if not df.empty else 0.0,
            "total_cost": float(df["cost"].sum()) if not df.empty else 0.0,
        }
        for m, df in diagnostics_dict.items()
    }
    with open(results_dir / "turnover_summary.json", "w") as f:
        json.dump(turnover_summary, f, indent=2)
    print("\nTurnover & cost summary:")
    print(pd.DataFrame(turnover_summary).T.round(4).to_string())

    # Stress test (2022)
    if port_returns.index.min() <= pd.Timestamp("2022-01-01") <= port_returns.index.max():
        stress = port_returns.loc["2022-01-01":"2022-12-31"]
        stress_summary = {m: summarize(stress[m].dropna()) for m in stress.columns}
        stress_df = pd.DataFrame(stress_summary).T.round(4)
        stress_df.to_csv(results_dir / "stress_test_2022.csv")
        print("\n=== 2022 stress test (high-vol regime) ===")
        print(stress_df.to_string())

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 7 — Plots")
    LOGGER.info("=" * 60)
    visualize.plot_equity_curves(equity, results_dir / "equity_curves.png")
    visualize.plot_drawdowns(port_returns, results_dir / "drawdowns.png")
    visualize.plot_weight_history(weights_dict, results_dir / "weight_history.png")
    visualize.plot_weight_stability(weights_dict, results_dir / "weight_stability.png")

    # Verify risk contributions for risk_parity at a sample date
    if "risk_parity" in weights_dict and not weights_dict["risk_parity"].empty:
        sample_date = weights_dict["risk_parity"].index[len(weights_dict["risk_parity"]) // 2]
        window_start = max(0, returns.index.get_loc(sample_date) - rb_cfg.lookback_days)
        sub_rets = returns.iloc[window_start : returns.index.get_loc(sample_date) + 1]
        cov_check, _ = ledoit_wolf_shrinkage(sub_rets, target="diag")
        w = weights_dict["risk_parity"].loc[sample_date].values
        rc = verify_risk_contributions(w, cov_check.values)
        LOGGER.info(
            "Risk Parity verification at %s: max deviation %.3f, equal within tol: %s",
            sample_date.date(), rc["max_deviation"], rc["is_equal_within_tol"],
        )
        with open(results_dir / "risk_parity_verification.json", "w") as f:
            json.dump(
                {
                    "date": sample_date.strftime("%Y-%m-%d"),
                    "max_deviation": rc["max_deviation"],
                    "is_equal_within_tol": rc["is_equal_within_tol"],
                    "rc_share": rc["rc_share"].tolist(),
                    "target_share": rc["target_share"].tolist(),
                    "tickers": list(cov_check.index),
                },
                f,
                indent=2,
            )

    LOGGER.info("Run complete. Outputs saved under %s", results_dir)

    # ------------------------------------------------------------------
    # 8. Robustness — sweep over (rebalance frequency × lookback)
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 8 — Robustness sweep (frequency × lookback)")
    LOGGER.info("=" * 60)
    sweep = robustness.run_robustness_sweep(
        returns,
        freq_grid=robustness.DEFAULT_FREQ_GRID,
        lookback_grid=robustness.DEFAULT_LOOKBACK_GRID,
        cost_bps=rb_cfg.cost_bps,
        cov_method=cfg["covariance"]["method"],
    )
    sweep.to_csv(results_dir / "robustness_sweep.csv", index=False)
    sweep_summary = robustness.summarize_sweep(sweep)
    sweep_summary.to_csv(results_dir / "robustness_summary.csv")
    print("\n=== Robustness sweep (mean ± std Sharpe across 6 configs) ===")
    print(sweep_summary.to_string())

    # ------------------------------------------------------------------
    # 9. Robustness — cost sensitivity
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 9 — Cost sensitivity")
    LOGGER.info("=" * 60)
    # Run with risk_overlay OFF so the cost effect is visible (vol-targeting
    # otherwise absorbs most of the cost by scaling exposure down).
    cost_df = robustness.run_cost_sensitivity(
        returns,
        cost_grid=robustness.DEFAULT_COST_GRID,
        frequency=rb_cfg.frequency,
        lookback_days=rb_cfg.lookback_days,
        cov_method=cfg["covariance"]["method"],
        risk_overlay=False,
    )
    cost_df.to_csv(results_dir / "cost_sensitivity.csv", index=False)
    cost_pivot = cost_df.pivot(index="cost_bps_roundtrip", columns="method", values="sharpe")
    cost_pivot.to_csv(results_dir / "cost_sensitivity_sharpe.csv")
    print("\n=== Cost sensitivity (Sharpe by round-trip cost, NO risk overlay) ===")
    print(cost_pivot.round(3).to_string())

    # ------------------------------------------------------------------
    # 10. Statistical significance for headline Sharpe
    # ------------------------------------------------------------------
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 10 — Sharpe statistical significance")
    LOGGER.info("=" * 60)
    sig = robustness.run_significance(port_returns, n_trials=5, n_boot=1000)
    sig.to_csv(results_dir / "sharpe_significance.csv")
    print("\n=== Sharpe significance (IID t-stat + 95% bootstrap CI) ===")
    print(sig.to_string())

    # Provenance log: currency mix, sources, date range
    provenance = {
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "data_sources": "vnstock(VCI) + yfinance HTTP",
        "date_range": f"{prices.index.min().date()} to {prices.index.max().date()}",
        "n_assets": int(prices.shape[1]),
        "n_obs": int(prices.shape[0]),
        "synthetic_fallback": bool(prices.attrs.get("currency_mix", {}).get("fx_applied", False) is False and "synthetic" in " ".join(sweep.columns.astype(str).tolist())),
        "currency_mix": prices.attrs.get("currency_mix", "synthetic (single currency)"),
    }
    with open(results_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portfolio Optimization Engine")
    p.add_argument(
        "--real-data",
        dest="synthetic",
        action="store_false",
        help="Fetch real market data via yfinance/vnstock (falls back to synthetic on failure).",
    )
    p.add_argument(
        "--synthetic",
        dest="synthetic",
        action="store_true",
        default=True,
        help="Use synthetic data (default; reproducible, no network needed).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())