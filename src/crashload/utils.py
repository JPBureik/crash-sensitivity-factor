# src/crashload/utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Date / stamping utilities
# ---------------------------

def coerce_date_series(dates: Iterable) -> pd.Series:
    """
    Return a Series of `datetime.date` (no tz) from many common inputs.
    """
    s = pd.to_datetime(pd.Series(dates), errors="coerce").dt.date
    if s.isna().any():
        bad = int(s.isna().sum())
        raise ValueError(f"Could not parse {bad} dates")
    return s


def month_ends_from_dates(dates: Iterable) -> pd.Series:
    """
    Given a collection of trading dates, return the last trading date for each month
    (based on the provided dates, not a holiday calendar).
    """
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    month_ends = d.groupby(d.dt.to_period("M")).max().dt.date
    return month_ends.sort_values().reset_index(drop=True)


def month_ends_from_returns(returns: pd.DataFrame, date_col: str = "date") -> pd.Series:
    """
    Derive trading month-ends from a panel of returns with a `date` column.
    """
    if date_col not in returns.columns:
        raise KeyError(f"returns is missing `{date_col}`")
    return month_ends_from_dates(returns[date_col])


def stamp_to_month_end(df: pd.DataFrame, date_col: str = "date", out_col: str = "month_end") -> pd.DataFrame:
    """
    Add a `month_end` column equal to the last trading day of the month for each row’s date.
    """
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    month_end_map = dates.groupby(dates.dt.to_period("M")).transform("max").dt.date
    out = df.copy()
    out[out_col] = month_end_map
    return out


# ---------------------------
# Winsorization / robust stats
# ---------------------------

def winsorize_series(
    s: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
    inclusive: bool = True,
) -> pd.Series:
    """
    Clip a Series by quantiles (symmetric if lower=1-upper).
    """
    if not 0 <= lower < upper <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    ql, qu = s.quantile([lower, upper])
    return s.clip(ql, qu) if inclusive else s.mask(s < ql, ql).mask(s > qu, qu)


def winsorize_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorize `value_col` within each group in `group_col`.
    """
    out = df.copy()
    out[value_col] = (
        out.groupby(group_col, sort=False)[value_col]
        .transform(lambda s: winsorize_series(s, lower, upper))
    )
    return out


def robust_center(s: pd.Series) -> float:
    """Median (robust mean) ignoring NaNs. Returns NaN if no finite values."""
    arr = np.asarray(s.values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def robust_scale_mad(s: pd.Series) -> float:
    """
    Robust std estimate via MAD (scaled by 1.4826).
    Returns:
      - NaN if no finite values
      - 0.0 if all finite values are equal
    """
    arr = np.asarray(s.values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return 1.4826 * mad


def robust_zscore(s: pd.Series) -> pd.Series:
    """
    Z-score using median/MAD.
    - If all values are NaN: return all NaN (no spurious zeros).
    - If scale==0: return zeros for finite entries and NaN where input is NaN.
    """
    arr = np.asarray(s.values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return pd.Series(np.nan, index=s.index, dtype=float)

    med = robust_center(s)
    scale = robust_scale_mad(s)

    if scale == 0 or not np.isfinite(scale):
        out = pd.Series(np.zeros(len(s), dtype=float), index=s.index)
        out[~finite] = np.nan
        return out

    return (s - med) / scale


# ---------------------------
# Simple shrinkage helpers
# ---------------------------

def ridge_shrink(value: float, n_eff: float, tau: float = 100.0, target: float = 0.0) -> float:
    """
    Ridge-style shrink toward `target` with strength `tau` using n_eff effective obs.
    weight = n_eff / (n_eff + tau)
    """
    w = n_eff / (n_eff + tau) if n_eff >= 0 else 0.0
    return float(w * value + (1 - w) * target)


def shrink_by_tstat(value: float, se: float, t_cap: float = 3.0, target: float = 0.0) -> float:
    """
    Shrink toward `target` so that |t| = |value/se| is capped by t_cap.
    If se<=0 or value==0, return target or value accordingly.
    """
    if se <= 0 or not np.isfinite(se):
        return float(target)
    t = abs(value / se)
    if not np.isfinite(t) or t <= t_cap:
        return float(value)
    # choose alpha in [0,1] so that |(1-alpha)*value| / se = t_cap
    alpha = max(0.0, min(1.0, 1.0 - t_cap / t))
    return float((1 - alpha) * value + alpha * target)


# ---------------------------
# Misc small helpers
# ---------------------------

@dataclass(frozen=True)
class WindowSpec:
    lookback_days: int
    min_obs: int = 250

def valid_rolling_mask(dates: Iterable, min_obs: int) -> pd.Series:
    """
    Given a date series, return a boolean mask True where count up to that index >= min_obs.
    """
    s = pd.Series(1, index=pd.to_datetime(pd.Series(dates)).index)
    c = s.cumsum()
    return c >= min_obs
