"""Data ingestion for the multi-asset portfolio.

Strategy:
1. Try fetching real OHLCV data via `yfinance` for the configured tickers
   (covers both VN tickers via `.VN` suffix and US ETFs/ETFs).
2. If `yfinance` is unavailable, the network fails, or no data is returned,
   fall back to a deterministic synthetic generator that mimics
   realistic multi-asset dynamics:
     - Sector-aware correlations (banks clustered, real-estate clustered, etc.)
     - Volatility regimes: low-vol (2018-2019), high-vol (2020 COVID, 2022),
       recovery (2023-2024).
     - Cross-asset decorrelation: bonds and gold behave distinctly from equities.

The synthetic generator uses a fixed seed so backtests are reproducible.

Outputs a single tidy parquet `data/processed/prices.parquet` with columns:
    date, ticker, open, high, low, close, adj_close, volume
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from src.config_loader import load_config, repo_path

LOGGER = logging.getLogger("ingestion")

# ---------------------------------------------------------------------------
# Real-data path (best-effort)
# ---------------------------------------------------------------------------

def _split_tickers(tickers: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    """Split a tickers list into (vn_tickers_raw, vn_tickers_yf, us_tickers).

    Vietnamese tickers are fetched via vnstock using their bare ticker (e.g. ``VNM``),
    while the same names also work on Yahoo Finance with the ``.VN`` suffix.
    We try vnstock first because it has cleaner data for VN equities.
    US-listed symbols (e.g. ``GLD``, ``TLT``) are fetched via yfinance.
    """
    vn_raw: list[str] = []
    vn_yf: list[str] = []
    us: list[str] = []
    for t in tickers:
        if t.endswith(".VN"):
            vn_raw.append(t.split(".")[0])
            vn_yf.append(t)
        else:
            us.append(t)
    return vn_raw, vn_yf, us


def _try_fetch_vnstock(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame | None:
    """Fetch Vietnamese equity adjusted close prices via ``vnstock``.

    Uses the vnstock 4.x ``Quote`` adapter with the VCI source. Returns a
    wide DataFrame (one column per ticker, ticker name without ``.VN``) or
    ``None`` on any failure.
    """
    try:
        from vnstock import Quote  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("vnstock not available (%s); skipping VN fetch.", exc)
        return None

    try:
        frames: dict[str, pd.Series] = {}
        for t in tickers:
            try:
                quote = Quote(source="VCI", symbol=t)
                df = quote.history(start=start, end=end, interval="1D")
            except Exception as exc:
                LOGGER.warning("vnstock fetch for %s failed (%s); skipping.", t, exc)
                continue
            if df is None or df.empty:
                continue
            # vnstock 4.x columns: time, open, high, low, close, volume
            date_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else df.columns[0])
            price_col = "close" if "close" in df.columns else df.columns[-1]
            s = pd.Series(df[price_col].values, index=pd.to_datetime(df[date_col]), name=t)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            frames[t] = s
        if not frames:
            return None
        wide = pd.concat(frames.values(), axis=1)
        wide.columns = list(frames.keys())
        return wide
    except Exception as exc:
        LOGGER.warning("vnstock top-level failure (%s); skipping VN fetch.", exc)
        return None


def _try_fetch_yfinance(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame | None:
    """Fetch US-listed adjusted close prices from Yahoo Finance via HTTP.

    We avoid importing the ``yfinance`` package (its ``protobuf`` extension
    is not compatible with Python 3.14) and call the public v8 chart API
    directly, which returns adjusted close prices in JSON form.
    """
    import json
    import time
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode

    tickers = list(tickers)
    if not tickers:
        return None

    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    frames: dict[str, pd.Series] = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }

    for t in tickers:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{t}?{urlencode({'period1': start_ts, 'period2': end_ts, 'interval': '1d', 'events': 'history'})}"
        )
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            LOGGER.warning("Yahoo HTTP fetch for %s failed (%s); skipping.", t, exc)
            continue

        try:
            result = payload["chart"]["result"][0]
            ts = result["timestamp"]
            indicators = result["indicators"]["quote"][0]
            closes = indicators.get("close") or indicators.get("adjclose", [{}])[0].get("adjclose")
            if not closes:
                continue
            idx = pd.to_datetime(ts, unit="s")
            s = pd.Series(closes, index=idx, name=t).dropna()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            frames[t] = s
        except (KeyError, IndexError, TypeError) as exc:
            LOGGER.warning("Yahoo parse for %s failed (%s); skipping.", t, exc)
            continue
        time.sleep(0.2)  # polite rate limit

    if not frames:
        return None
    wide = pd.concat(frames.values(), axis=1)
    wide.columns = list(frames.keys())
    return wide


def fetch_fx_rate(pair: str = "USDVND=X", start: str = "2017-01-01",
                  end: str = "2024-12-31") -> pd.Series | None:
    """Fetch a daily FX rate (e.g. USDVND=X) from Yahoo Finance.

    Returns a pandas Series indexed by date, or ``None`` on failure.
    Used by ``convert_to_base_currency`` to merge USD-denominated assets
    into a VND-base portfolio.
    """
    df = _try_fetch_yfinance([pair], start, end)
    if df is None or df.empty:
        return None
    return df[pair]


def convert_us_to_vnd(prices: pd.DataFrame, us_tickers: list[str],
                      fx_pair: str = "USDVND=X") -> pd.DataFrame:
    """Multiply US-denominated price columns by the appropriate FX rate.

    Converts each column in ``us_tickers`` from USD to VND using
    ``fx_pair`` (default ``USDVND=X``, i.e. number of VND per 1 USD).
    Returns the price DataFrame with the same shape, in-place update.
    """
    fx = fetch_fx_rate(fx_pair)
    if fx is None:
        raise RuntimeError(
            f"Could not fetch FX rate {fx_pair}. Cannot convert USD prices to VND."
        )
    out = prices.copy()
    # Reindex FX onto the price index, forward-fill up to 3 days (weekends)
    fx_aligned = fx.reindex(out.index).ffill(limit=3)
    for t in us_tickers:
        if t in out.columns:
            out[t] = out[t] * fx_aligned
            LOGGER.info("Converted %s to VND (1 USD ≈ %.0f VND avg)", t, fx_aligned.mean())
    out.attrs["fx_applied"] = {fx_pair: "USDVND", "tickers": us_tickers}
    return out


# ---------------------------------------------------------------------------
# Synthetic data generator (fallback, fully reproducible)
# ---------------------------------------------------------------------------

# Map tickers to sectors used to drive correlation structure
_SECTOR_MAP = {
    "VNM.VN": "consumer", "VIC.VN": "realestate", "VHM.VN": "realestate",
    "HPG.VN": "industrials", "BID.VN": "banks", "CTG.VN": "banks",
    "VCB.VN": "banks", "TCB.VN": "banks", "MBB.VN": "banks",
    "FPT.VN": "tech",
    "GLD": "gold", "TLT": "bonds",
}

# Asset-level annualized expected return and volatility targets (deterministic)
_ASSET_PROFILE = {
    "VNM.VN": (0.10, 0.22), "VIC.VN": (0.06, 0.40), "VHM.VN": (0.05, 0.42),
    "HPG.VN": (0.12, 0.38), "BID.VN": (0.10, 0.32), "CTG.VN": (0.10, 0.30),
    "VCB.VN": (0.11, 0.28), "TCB.VN": (0.14, 0.36), "MBB.VN": (0.12, 0.34),
    "FPT.VN": (0.18, 0.32),
    "GLD": (0.05, 0.15), "TLT": (0.03, 0.12),
}

# Within-sector base correlation; cross-sector lower; gold/bonds anti-correlated with equities in stress
_INTRA_SECTOR_CORR = 0.55
_INTER_SECTOR_CORR = 0.25
_BOND_EQUITY_BASE_CORR = -0.05
_GOLD_EQUITY_BASE_CORR = 0.05


def _build_correlation_matrix(tickers: List[str]) -> np.ndarray:
    """Build a sector-aware correlation matrix for the universe."""
    n = len(tickers)
    corr = np.full((n, n), _INTER_SECTOR_CORR, dtype=float)
    sectors = [_SECTOR_MAP.get(t, "other") for t in tickers]
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
                continue
            si, sj = sectors[i], sectors[j]
            if si == sj:
                corr[i, j] = _INTRA_SECTOR_CORR
            # Bonds are weakly negatively correlated with equities in this stylized setup
            if {si, sj} == {"bonds", "banks"} or {si, sj} == {"bonds", "industrials"}:
                corr[i, j] = _BOND_EQUITY_BASE_CORR
            if "gold" in (si, sj) and sj != si:
                corr[i, j] = _GOLD_EQUITY_BASE_CORR
    # Symmetrize (already symmetric by construction)
    return corr


def _build_volatility_regime(index: pd.DatetimeIndex) -> np.ndarray:
    """Return a multiplier (~0.7 calm to ~1.8 stress) for daily vol at each timestamp.

    Captures: COVID crash (Q1 2020), 2022 inflation/rate-hike stress, calm periods.
    """
    n = len(index)
    rng = np.random.default_rng(seed=42)  # deterministic per-row noise
    base = np.ones(n)
    dates = index.values
    for i, d in enumerate(dates):
        year = pd.Timestamp(d).year
        month = pd.Timestamp(d).month
        if year == 2020 and month <= 6:
            base[i] = 1.8  # COVID shock
        elif year == 2022:
            base[i] = 1.3  # inflation/rate hike stress
        elif year == 2018 and month >= 10:
            base[i] = 1.2  # Q4 2018 selloff
        elif year == 2023:
            base[i] = 0.85  # recovery calmer
        elif year == 2024:
            base[i] = 0.8   # very calm
        else:
            base[i] = 1.0
    # Light temporal smoothing so regime changes aren't jarring
    kernel = np.ones(10) / 10
    smoothed = np.convolve(base, kernel, mode="same")
    # Add small idiosyncratic noise
    smoothed = smoothed * (1 + rng.normal(0, 0.02, size=n))
    return np.clip(smoothed, 0.5, 2.5)


def _generate_synthetic_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Generate a wide DataFrame of synthetic adjusted close prices.

    Uses Cholesky-decomposed multivariate normal returns with regime-aware
    volatility scaling. Reproducible via fixed seed.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    # Use NYSE-style business days; sufficient for our purposes
    index = pd.bdate_range(start=start_ts, end=end_ts)
    n_obs = len(index)
    n_assets = len(tickers)

    annual_returns = np.array([_ASSET_PROFILE.get(t, (0.08, 0.25))[0] for t in tickers])
    annual_vols = np.array([_ASSET_PROFILE.get(t, (0.08, 0.25))[1] for t in tickers])
    daily_mu = annual_returns / 252.0
    daily_vol = annual_vols / np.sqrt(252.0)

    corr = _build_correlation_matrix(tickers)
    # Ensure positive semi-definiteness via tiny jitter
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 0:
        corr = corr + (-eigvals.min() + 1e-6) * np.eye(n_assets)
    L = np.linalg.cholesky(corr)

    regime = _build_volatility_regime(index)  # length n_obs
    rng = np.random.default_rng(seed=20240101)
    shocks = rng.standard_normal((n_obs, n_assets))
    correlated = shocks @ L.T
    scaled = correlated * daily_vol[None, :] * regime[:, None] + daily_mu[None, :]
    # Convert log-returns approximated by simple returns; cumulative product
    price_paths = 100.0 * np.exp(np.cumsum(scaled, axis=0))
    prices = pd.DataFrame(price_paths, index=index, columns=tickers)
    prices.index.name = "date"
    return prices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_prices(cfg: dict | None = None, use_synthetic: bool = False) -> pd.DataFrame:
    """Fetch (or generate) a wide DataFrame of adjusted close prices.

    Parameters
    ----------
    cfg : dict, optional
        Config dict; loaded automatically if omitted.
    use_synthetic : bool
        If True, skip real-data fetch entirely and use synthetic data.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (business days). Columns: tickers. Values: prices.
    """
    cfg = cfg or load_config()
    tickers = list(cfg["universe"]["vn30"]) + list(cfg["universe"]["diversification"])
    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]

    vn_raw, vn_yf, us = _split_tickers(tickers)

    prices: pd.DataFrame | None = None
    source_used: list[str] = []
    if not use_synthetic:
        # Step 1: VN tickers via vnstock (preferred), fallback to yfinance .VN
        vn_prices = _try_fetch_vnstock(vn_raw, start, end) if vn_raw else None
        if vn_prices is None and vn_yf:
            LOGGER.info("Falling back to yfinance for VN tickers (%s)", vn_yf)
            vn_prices = _try_fetch_yfinance(vn_yf, start, end)
            if vn_prices is not None:
                # Normalize column names: drop .VN suffix to match vnstock output
                vn_prices = vn_prices.rename(columns={c: c.split(".")[0] for c in vn_prices.columns})
        if vn_prices is not None and not vn_prices.empty:
            source_used.append(f"vnstock({len(vn_prices.columns)})" if _try_fetch_vnstock is not None else "yfinance(.VN)")
            prices = vn_prices
        # Step 2: US tickers via yfinance
        if us:
            us_prices = _try_fetch_yfinance(us, start, end)
            if us_prices is not None and not us_prices.empty:
                source_used.append(f"yfinance({','.join(us_prices.columns)})")
                prices = us_prices if prices is None else prices.join(us_prices, how="outer")

    if prices is None or prices.empty:
        LOGGER.info("Using synthetic data for %d tickers from %s to %s.", len(tickers), start, end)
        prices = _generate_synthetic_prices(tickers, start, end)
        source_used.append("synthetic")
        currency_mix = {"base": "USD-equivalent", "fx_applied": False,
                        "note": "all assets in same currency (synthetic generator)"}
    else:
        # Restore the original column order (with .VN suffix) and ensure each requested ticker is present
        # If a ticker failed to download in any source, fill it with synthetic data and warn
        missing = [t for t in tickers if t.split(".")[0] not in prices.columns and t not in prices.columns]
        if missing:
            LOGGER.warning("Missing tickers in real-data fetch %s; filling with synthetic.", missing)
            synth = _generate_synthetic_prices(missing, start, end)
            prices = prices.join(synth, how="outer")
        # Rename bare VN tickers back to .VN form to match the original config
        prices = prices.rename(columns={c: f"{c}.VN" for c in vn_raw if c in prices.columns and f"{c}.VN" not in prices.columns})

        # FX handling: VN prices are in thousands of VND; GLD/TLT are in USD.
        # We do NOT convert them here — we leave prices in their native currency
        # and tag the column with a metadata dict in `prices.attrs['currency']`
        # so the backtest and the user are aware of the cross-currency exposure.
        # See RESEARCH_NOTE.md → Limitations for the implications.
        currency_mix = {
            "vn_tickers": [f"{t}.VN" for t in vn_raw],
            "us_tickers": list(us),
            "vn_unit": "1000 VND (unadjusted close)",
            "us_unit": "USD",
            "fx_applied": False,
            "note": (
                "Returns are computed in each asset's NATIVE currency. The portfolio "
                "sum-of-weighted-returns implicitly assumes a constant VND/USD rate. "
                "In reality, VND has depreciated ~2-3% per year vs USD over 2017-2024, "
                "which mechanically inflates USD-asset (GLD/TLT) returns for a VND-base "
                "investor by that amount. We do NOT correct for this — see Limitations."
            ),
        }
        prices.attrs["currency_mix"] = currency_mix

    LOGGER.info("Data sources used: %s", ", ".join(source_used))
    if "currency_mix" in dir():
        LOGGER.info("Currency mix: %s", currency_mix)

    # Ensure column order matches config
    prices = prices[[t for t in tickers if t in prices.columns]]
    prices = prices.sort_index()
    return prices


def align_calendar(prices: pd.DataFrame, max_ffill: int = 3) -> pd.DataFrame:
    """Forward-fill missing values up to `max_ffill` consecutive days.

    For multi-asset portfolios spanning different exchanges, some dates may
    be missing for some tickers (e.g. US holidays when VN markets were open).
    NaNs beyond the threshold are left in place and dropped downstream.
    """
    return prices.ffill(limit=max_ffill)


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute periodic returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
    method : {"log", "simple"}
    """
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}")
    return rets.dropna(how="all")


def save_processed(prices: pd.DataFrame, returns: pd.DataFrame, out_dir: Path | None = None) -> dict:
    """Persist processed prices and returns to parquet."""
    out_dir = out_dir or repo_path("data", "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    prices_path = out_dir / "prices.parquet"
    returns_path = out_dir / "returns.parquet"
    prices.to_parquet(prices_path)
    returns.to_parquet(returns_path)
    LOGGER.info("Wrote %s (%d rows) and %s (%d rows)", prices_path, len(prices), returns_path, len(returns))
    return {"prices": prices_path, "returns": returns_path}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    cfg = load_config()
    prices = fetch_prices(cfg)
    prices = align_calendar(prices)
    returns = compute_returns(prices, method="log")
    save_processed(prices, returns)


if __name__ == "__main__":
    main()