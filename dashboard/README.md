# Dashboard

Streamlit dashboard for the backtest engine.

The dashboard is **read-only** — it never runs the backtest itself. It
just reads the pre-computed artefacts in `../results/` (produced by
`run.py`) and renders them with Plotly. That keeps the Space
CPU-only-friendly and the page snappy.

## Layout

```
dashboard/
├── data_loader.py     # Cached CSV/JSON readers
├── charts.py          # Plotly figure builders
└── tabs/
    ├── overview.py    # KPIs + equity + drawdown + correlation
    ├── comparison.py  # Metric table, per-metric bar, risk/return scatter
    ├── weights.py     # Stacked area + turnover + risk-contribution check
    └── robustness.py  # Sweep + cost sensitivity + bootstrap CI
```

`app.py` at the repo root is the Streamlit entrypoint. It assembles the
sidebar (method multi-select, date-range slider) and delegates to the
four tab modules.

## Local run

```bash
pip install -r requirements-dashboard.txt
streamlit run app.py
```

## HF Spaces deploy

1. Create a new Space with `sdk: streamlit`.
2. Push this repo (or a copy) to the Space.
3. Add the YAML front-matter (title, sdk_version, app_file) to the Space's
   `README.md` — see the top of the root `README.md` for the snippet.
4. The Space starts on `app.py` and reads `results/` from the repo.

If the Space repo only contains `dashboard/` + `app.py` (no `run.py` /
`results/`), the dashboard will render but most charts will be empty.
Commit a fresh `results/` after each `python run.py --real-data`.
