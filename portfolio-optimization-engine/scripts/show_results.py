"""Print a one-page summary of the most recent backtest run.

Reads the artifacts produced by ``run.py`` under ``results/`` and prints a
compact table comparing every method on the headline risk/return metrics.

Usage::

    python scripts/show_results.py
    python scripts/show_results.py --stress   # also include the 2022 stress slice
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

METRIC_LABELS = {
    "annualized_return": "Ann.Ret",
    "annualized_volatility": "Ann.Vol",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "max_drawdown": "Max DD",
    "effective_n_avg": "Eff.N",
    "mean_weight_change": "Mean |dW|",
    "diversification_ratio": "Div.Ratio",
}


def _format_table(df: pd.DataFrame, title: str) -> str:
    pretty = df.rename(columns=METRIC_LABELS)
    line = "=" * 78
    out = [line, title.center(78), line, pretty.to_string(float_format=lambda x: f"{x:>8.3f}"), line]
    return "\n".join(out)


def main(args: argparse.Namespace) -> None:
    comp_path = RESULTS / "comparison_table.csv"
    if not comp_path.exists():
        raise SystemExit(
            "No comparison_table.csv found. Run `python run.py` first."
        )

    comp = pd.read_csv(comp_path, index_col=0)
    print(_format_table(comp, "BACKTEST COMPARISON (full sample)"))

    turnover_path = RESULTS / "turnover_summary.json"
    if turnover_path.exists():
        turnover = pd.DataFrame(json.loads(turnover_path.read_text())).T
        print("\n" + _format_table(turnover, "TURNOVER & COST SUMMARY"))

    if args.stress:
        stress_path = RESULTS / "stress_test_2022.csv"
        if stress_path.exists():
            stress = pd.read_csv(stress_path, index_col=0)
            print("\n" + _format_table(stress, "2022 STRESS TEST"))
        else:
            print("\n[2022 stress slice not in sample range — skipping]")

    print("\nArtefacts available under:", RESULTS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize the latest backtest results.")
    p.add_argument("--stress", action="store_true", help="Also show the 2022 stress slice.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())