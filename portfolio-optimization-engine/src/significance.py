"""Statistical significance for the Sharpe ratio.

Implements:
- **t-statistic** under the (standard but incorrect) IID normality assumption
  (Lo 2002). Reported for context only — always pair with bootstrap.
- **Bootstrap CI** (circular block bootstrap with block length ~sqrt(T)) for
  the Sharpe ratio. Returns 95% CI by default. This is the canonical
  non-parametric CI cited in the literature (Politis & Romano 1994;
  Ledoit & Wolf 2008).

Also implements:
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014): adjusts Sharpe
  for the number of trials / multiple testing. ``n_trials=1`` reduces to
  the standard SR.

Both functions are pure numpy/pandas; no engine state required.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _ann_sharpe_from_daily(daily: np.ndarray, periods_per_year: int = 252) -> float:
    if daily.size < 2:
        return float("nan")
    excess = daily
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sharpe_t_stat(returns: pd.Series, periods_per_year: int = 252) -> Tuple[float, float, float]:
    """t-statistic for the Sharpe ratio under IID normality.

    Returns
    -------
    (t_stat, p_value_two_sided, sharpe)
    """
    from scipy import stats

    r = returns.dropna().values
    n = len(r)
    if n < 3:
        return (float("nan"), float("nan"), float("nan"))
    sr = _ann_sharpe_from_daily(r, periods_per_year)
    # Lo (2002): var(SR) ≈ (1 + 0.5 SR^2) / (T - 1)
    var_sr = (1.0 + 0.5 * sr ** 2) / max(n - 1, 1)
    se = np.sqrt(var_sr)
    if se == 0 or np.isnan(se):
        return (float("nan"), float("nan"), sr)
    t_stat = sr / se
    # two-sided p-value (Student-t with df = n-1)
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1))
    return (float(t_stat), float(p_value), float(sr))


def circular_block_bootstrap_sharpe(
    returns: pd.Series,
    n_boot: int = 1000,
    block_size: int | None = None,
    seed: int = 42,
    ci: float = 0.95,
    periods_per_year: int = 252,
) -> dict:
    """Circular block bootstrap for the annualized Sharpe ratio.

    Block length follows Politis & Romano (1994) automatic rule of thumb:
    ``block = ceil(T ** (1/3))``. We round that to a small integer which
    gives stable SR estimates on daily data with T ~ 2000.

    Returns
    -------
    dict with keys ``sharpe``, ``se``, ``ci_low``, ``ci_high``, ``n_boot``,
    ``block_size``.
    """
    r = returns.dropna().values
    n = len(r)
    if n < 60:
        return {"sharpe": float("nan"), "se": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot": n_boot, "block_size": 0}

    if block_size is None:
        block_size = max(5, int(np.ceil(n ** (1.0 / 3.0))))

    rng = np.random.default_rng(seed)
    boot_sr = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        # Circular block bootstrap: sample block_size starting positions with
        # wrap-around, then concatenate to length n.
        starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
        idx = np.empty(int(np.ceil(n / block_size)) * block_size, dtype=int)
        for j, s in enumerate(starts):
            block = (s + np.arange(block_size)) % n
            idx[j * block_size:(j + 1) * block_size] = block
        idx = idx[:n]
        sample = r[idx]
        boot_sr[b] = _ann_sharpe_from_daily(sample, periods_per_year)

    # Drop NaN / inf from degenerate samples
    boot_sr = boot_sr[np.isfinite(boot_sr)]
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.quantile(boot_sr, alpha))
    ci_high = float(np.quantile(boot_sr, 1.0 - alpha))
    point = _ann_sharpe_from_daily(r, periods_per_year)
    return {
        "sharpe": float(point),
        "se": float(boot_sr.std(ddof=1)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": int(len(boot_sr)),
        "block_size": int(block_size),
        "ci_level": ci,
    }


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int = 1,
    periods_per_year: int = 252,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Adjusts the observed Sharpe for multiple-testing bias. ``n_trials=1``
    reduces to the standard Sharpe.

    Returns
    -------
    dict with ``sharpe``, ``deflated_sharpe`` (E[max SR under null]),
    ``prob_sharpe_underestimated``, ``sharpe_threshold``.
    """
    r = returns.dropna().values
    n = len(r)
    if n < 30:
        return {"sharpe": float("nan"), "deflated_sharpe": float("nan"),
                "prob_sharpe_underestimated": float("nan"),
                "sharpe_threshold": float("nan")}

    sr = _ann_sharpe_from_daily(r, periods_per_year)
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis())  # excess kurtosis

    # E[max SR under null] for n_trials independent normals
    e_max = _expected_max_sharpe(n_trials, n)
    if n_trials <= 1 or e_max <= 0:
        sr_star = float("nan")
        prob_underest = float("nan")
    else:
        # SR* threshold (the expected maximum SR under the null of zero true SR)
        # Bailey & Lopez de Prado (2014), Eq. 3
        inner = (1.0 - e_max ** (-1) * (1.0 - e_max ** (-1)))
        inner = max(inner, 0.0)
        sigma_z = np.sqrt(
            inner * (1.0 + 0.5 * sr ** 2 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
            / (n - 1)
        )
        sr_star = float(np.sqrt(n - 1) * sigma_z) if not np.isnan(sigma_z) else float("nan")
        # Probability the observed SR exceeds SR* (one-sided test)
        var_sr = (1.0 + 0.5 * sr ** 2 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n - 1)
        sigma_sr = np.sqrt(max(var_sr, 1e-12))
        if not np.isnan(sr_star):
            z = (sr - sr_star) / sigma_sr
            from scipy import stats
            prob_underest = float(stats.norm.cdf(z))
        else:
            prob_underest = float("nan")

    return {
        "sharpe": float(sr),
        "deflated_sharpe": float(e_max),
        "prob_sharpe_underestimated": prob_underest,
        "sharpe_threshold": sr_star,
        "n_trials": int(n_trials),
        "n_obs": int(n),
        "skewness": skew,
        "excess_kurtosis": kurt,
    }


def _expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """E[max of n_trials IID standard normals] ≈
    (1 - γ) Φ^{-1}(1 - 1/n) + γ Φ^{-1}(1 - 1/(n e))  (BLP 2014 eq. 4)
    """
    from scipy import stats

    if n_trials <= 1:
        return 0.0
    euler_gamma = 0.5772156649
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float((1.0 - euler_gamma) * z1 + euler_gamma * z2)


__all__ = [
    "sharpe_t_stat",
    "circular_block_bootstrap_sharpe",
    "deflated_sharpe",
]