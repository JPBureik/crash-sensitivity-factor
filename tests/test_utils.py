# tests/test_utils.py
import dataclasses
import numpy as np
import pandas as pd
import pytest

from crashload.utils import (
    month_ends_from_dates,
    winsorize_series,
    robust_center,
    robust_scale_mad,
    coerce_date_series,
    month_ends_from_returns,
    stamp_to_month_end,
    winsorize_by_group,
    robust_scale_mad,
    robust_zscore,
    ridge_shrink,
    shrink_by_tstat,
    valid_rolling_mask,
    WindowSpec,
)

def test_month_ends_from_dates_basic():
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="B").date
    me = month_ends_from_dates(dates)
    assert list(me) == [pd.to_datetime("2024-01-31").date(),
                        pd.to_datetime("2024-02-29").date(),
                        pd.to_datetime("2024-03-29").date()]

def test_winsorize_series_bounds():
    s = pd.Series([-100, -1, 0, 1, 100])
    w = winsorize_series(s, 0.2, 0.8)
    assert w.min() >= s.quantile(0.2) - 1e-12
    assert w.max() <= s.quantile(0.8) + 1e-12

def test_robust_center_scale_with_outliers():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=1000)
    x[:10] = 20  # big outliers
    s = pd.Series(x)
    med = robust_center(s)
    sc = robust_scale_mad(s)
    assert abs(med) < 0.2
    assert 0.5 < sc < 2.5  # robust to outliers

def test_robust_zscore_zero_when_constant():
    s = pd.Series([5, 5, 5])
    z = robust_zscore(s)
    assert (z == 0).all()

def test_ridge_shrink_toward_zero():
    v = 1.0
    shrunk = ridge_shrink(v, n_eff=100, tau=100)  # weight = 0.5
    assert abs(shrunk - 0.5) < 1e-9

def test_shrink_by_tstat_caps_t():
    v, se, tcap = 10.0, 1.0, 2.0  # t=10
    shrunk = shrink_by_tstat(v, se, tcap)
    # result should have |value/se| <= tcap
    assert abs(shrunk / se) <= tcap + 1e-9

def test_coerce_date_series_ok_and_bad():
    ok = coerce_date_series(["2024-01-01", "2024-01-02"])
    assert ok.dtype == "object" and all(hasattr(d, "year") for d in ok)

    with pytest.raises(ValueError):
        coerce_date_series(["2024-01-01", "not-a-date"])

def test_month_ends_from_returns_and_stamp_to_month_end():
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", "2024-03-31"),
        "ret": np.arange(65) * 0.0
    })
    me = month_ends_from_returns(df)  # uses default date_col
    assert list(me)[-1].month == 3

    stamped = stamp_to_month_end(df, date_col="date", out_col="month_end")
    g = stamped.groupby(stamped["date"].dt.to_period("M"))["date"].max().dt.date
    # every row’s month_end equals that month’s last trading day
    assert (stamped["month_end"].isin(g.values)).all()

def test_winsorize_by_group():
    df = pd.DataFrame({
        "grp": ["A"]*5 + ["B"]*5,
        "x":   [-100, -1, 0, 1, 100, -200, -2, 0, 2, 200],
    })
    out = winsorize_by_group(df, value_col="x", group_col="grp", lower=0.2, upper=0.8)
    a = out.loc[out["grp"]=="A","x"]
    b = out.loc[out["grp"]=="B","x"]
    # groupwise clipping happened (extremes pulled toward group quantiles)
    assert a.min() >= df.loc[df["grp"]=="A","x"].quantile(0.2) - 1e-12
    assert a.max() <= df.loc[df["grp"]=="A","x"].quantile(0.8) + 1e-12
    assert b.min() >= df.loc[df["grp"]=="B","x"].quantile(0.2) - 1e-12
    assert b.max() <= df.loc[df["grp"]=="B","x"].quantile(0.8) + 1e-12

def test_robust_scale_mad_constant_and_zscore_nonconstant():
    const = pd.Series([5.0, 5.0, 5.0])
    assert robust_scale_mad(const) == 0.0

    s = pd.Series([0.0, 1.0, 2.0])
    z = robust_zscore(s)
    # median is 1, MAD>0, so z has negative/zero/positive values
    assert z.iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert z.iloc[0] < 0 and z.iloc[2] > 0

def test_ridge_shrink_edge_cases():
    # negative n_eff → weight=0 -> return target
    assert ridge_shrink(10.0, n_eff=-5, tau=100.0, target=1.23) == pytest.approx(1.23)
    # finite tau: compare to the formula, not to 2.0 exactly
    v, n_eff, tau, target = 2.0, 1_000_000_000, 100.0, -99.0
    shrunk = ridge_shrink(v, n_eff=n_eff, tau=tau, target=target)
    w = n_eff / (n_eff + tau)
    expected = w * v + (1 - w) * target
    assert shrunk == pytest.approx(expected, rel=1e-12)
    # sanity: it's very close to v
    assert abs(shrunk - v) < 2e-5


def test_shrink_by_tstat_edge_cases():
    # se<=0 → return target
    assert shrink_by_tstat(10.0, se=0.0, t_cap=2.0, target=-1.0) == pytest.approx(-1.0)
    # value==0 → returns value (0), regardless of se, since |t|=0 <= t_cap
    assert shrink_by_tstat(0.0, se=1.0, t_cap=2.0, target=5.0) == pytest.approx(0.0)
    # very large t shrinks to cap
    out = shrink_by_tstat(10.0, se=1.0, t_cap=2.0, target=0.0)
    assert abs(out/1.0) <= 2.0 + 1e-12

def test_valid_rolling_mask():
    dates = pd.bdate_range("2024-01-01", periods=10)
    mask = valid_rolling_mask(dates, min_obs=5)
    # first 4 are False, from 5th onward True
    assert mask.sum() == 6 and mask.iloc[4] and not mask.iloc[3]

def test_window_spec_is_frozen():
    ws = WindowSpec(lookback_days=252, min_obs=200)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ws.min_obs = 123

def test_month_ends_from_returns_missing_date_col_raises():
    df = pd.DataFrame({"when": pd.bdate_range("2024-01-01", periods=5)})
    with pytest.raises(KeyError, match="date"):
        month_ends_from_returns(df)  # default date_col='date'

def test_winsorize_series_invalid_quantiles_raises():
    s = pd.Series([1, 2, 3])
    # lower >= upper
    with pytest.raises(ValueError):
        winsorize_series(s, 0.8, 0.2)
    with pytest.raises(ValueError):
        winsorize_series(s, 0.5, 0.5)
    # out of [0,1]
    with pytest.raises(ValueError):
        winsorize_series(s, -0.1, 0.9)
    with pytest.raises(ValueError):
        winsorize_series(s, 0.1, 1.1)