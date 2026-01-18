import numpy as np
import pandas as pd
import pytest

from crashload.estimators import (
    CoMomentConfig,
    _demean,
    cokurt_beta,
    coskew_beta,
    joint_cumulant3,
    joint_cumulant4,
    panel_betas,
    rolling_beta_series,
)


def test_gaussian_cobet_asymptotically_zero():
    rng = np.random.default_rng(0)
    n = 20000
    rm = pd.Series(rng.normal(0, 1, n))
    # independent stock
    ri = pd.Series(rng.normal(0, 1, n))
    cfg = CoMomentConfig(winsor=0.0, robust_scale=False, shrink_tau=0.0)
    b3 = coskew_beta(ri, rm, cfg)
    b4 = cokurt_beta(ri, rm, cfg)
    assert abs(b3) < 0.03
    assert abs(b4) < 0.05


def test_coskew_positive_when_ri_tracks_rm_squared():
    rng = np.random.default_rng(1)
    n = 5000
    rm = pd.Series(rng.normal(0, 1, n))
    # make ri ~ 0.5 * (rm^2 - 1) + noise  (centered quadratic -> positive coskew)
    ri = pd.Series(0.5 * (rm**2 - 1) + 0.1 * rng.normal(0, 1, n))
    cfg = CoMomentConfig(winsor=0.0, robust_scale=False, shrink_tau=0.0)
    b3 = coskew_beta(ri, rm, cfg)
    assert b3 > 0.1


def test_cokurt_positive_when_ri_tracks_rm_cubed():
    rng = np.random.default_rng(2)
    n = 5000
    rm = pd.Series(rng.normal(0, 1, n))
    ri = pd.Series(0.3 * (rm**3) + 0.1 * rng.normal(0, 1, n))
    cfg = CoMomentConfig(winsor=0.0, robust_scale=False, shrink_tau=0.0)
    b4 = cokurt_beta(ri, rm, cfg)
    assert b4 > 0.1


# ---- joint cumulants: empty + center=False path ----


def test_joint_cumulant3_empty_returns_nan():
    x = pd.Series([np.nan, np.nan])
    y = pd.Series([np.nan, np.nan])
    z = pd.Series([np.nan, np.nan])
    assert np.isnan(joint_cumulant3(x, y, z, center=True))
    assert np.isnan(joint_cumulant3(x, y, z, center=False))


def test_joint_cumulant3_general_formula_matches_manual():
    x = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series([0.5, 1.0, 1.5])
    z = pd.Series([2.0, 0.0, -1.0])
    # manual general formula
    EX, EY, EZ = x.mean(), y.mean(), z.mean()
    EXY, EXZ, EYZ = (x * y).mean(), (x * z).mean(), (y * z).mean()
    EXYZ = (x * y * z).mean()
    expected = EXYZ - EX * EYZ - EY * EXZ - EZ * EXY + 2 * EX * EY * EZ
    got = joint_cumulant3(x, y, z, center=False)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_joint_cumulant4_center_true_and_general():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=500))
    b = pd.Series(rng.normal(size=500))
    c = pd.Series(rng.normal(size=500))
    d = pd.Series(rng.normal(size=500))

    # center=True: κ4 = E[abcd] - E[ab]E[cd] - E[ac]E[bd] - E[ad]E[bc]
    def E(s: pd.Series) -> float:
        return float(s.mean())

    A, B, C, D = a - a.mean(), b - b.mean(), c - c.mean(), d - d.mean()
    expected_center = (
        E(A * B * C * D)
        - E(A * B) * E(C * D)
        - E(A * C) * E(B * D)
        - E(A * D) * E(B * C)
    )
    got_center = joint_cumulant4(a, b, c, d, center=True)
    assert got_center == pytest.approx(expected_center, rel=1e-12, abs=1e-12)

    # center=False: just make sure it runs and returns a finite float
    got_general = joint_cumulant4(a, b, c, d, center=False)
    assert np.isfinite(got_general)


# ---- coskew/cokurt: denom=0 and shrink paths + robust/winsor branches ----


def test_coskew_and_cokurt_denominator_zero_returns_nan():
    # rm constant -> var=0 => NaN
    rm = pd.Series([1.0] * 100)
    ri = pd.Series(np.linspace(-1, 1, 100))
    cfg = CoMomentConfig(winsor=0.0, robust_scale=True, shrink_tau=0.0)
    assert np.isnan(coskew_beta(ri, rm, cfg))
    assert np.isnan(cokurt_beta(ri, rm, cfg))


def test_coskew_and_cokurt_with_winsor_scale_and_shrink():
    rng = np.random.default_rng(42)
    n = 5000
    rm = pd.Series(rng.normal(0, 1, n))
    # induce nonzero co-moments
    ri3 = pd.Series(0.4 * (rm**2 - 1) + 0.2 * rng.normal(0, 1, n))  # coskew-ish
    ri4 = pd.Series(0.2 * (rm**3) + 0.2 * rng.normal(0, 1, n))  # cokurt-ish

    cfg_no_shrink = CoMomentConfig(winsor=0.05, robust_scale=True, shrink_tau=0.0)
    cfg_shrink = CoMomentConfig(winsor=0.05, robust_scale=True, shrink_tau=200.0)

    b3_nos = coskew_beta(ri3, rm, cfg_no_shrink)
    b3_shr = coskew_beta(ri3, rm, cfg_shrink)
    assert abs(b3_shr) < abs(b3_nos)  # shrink pulls toward 0

    b4_nos = cokurt_beta(ri4, rm, cfg_no_shrink)
    b4_shr = cokurt_beta(ri4, rm, cfg_shrink)
    assert abs(b4_shr) < abs(b4_nos)


def test_coskew_and_cokurt_empty_after_dropna():
    # All NaNs -> _prep_pair returns empty series -> NaN result
    ri = pd.Series([np.nan, np.nan])
    rm = pd.Series([np.nan, np.nan])
    assert np.isnan(coskew_beta(ri, rm))
    assert np.isnan(cokurt_beta(ri, rm))


# ---- rolling & panel betas ----


def test_rolling_beta_series_empty_join_returns_nan_index():
    # rm all NaN => joined empty; function should return Series aligned to ri index
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    ri = pd.Series([1, 2, 3, 4, 5], index=idx, dtype=float)
    rm = pd.Series([np.nan] * 5, index=idx, dtype=float)
    out = rolling_beta_series(ri, rm, window=3, min_obs=2, which="coskew")
    assert out.index.equals(ri.index)
    assert out.isna().all()


def test_rolling_beta_series_basic_for_both_metrics():
    rng = np.random.default_rng(123)
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    rm = pd.Series(rng.normal(size=len(idx)), index=idx)
    ri = pd.Series(
        0.2 * (rm**2 - rm.mean() ** 2) + 0.1 * rng.normal(size=len(idx)), index=idx
    )

    # coskew
    out3 = rolling_beta_series(
        ri,
        rm,
        window=10,
        min_obs=5,
        which="coskew",
        cfg=CoMomentConfig(winsor=0.0, robust_scale=False, shrink_tau=0.0),
    )
    # cokurt
    out4 = rolling_beta_series(
        ri,
        rm,
        window=10,
        min_obs=5,
        which="cokurt",
        cfg=CoMomentConfig(winsor=0.0, robust_scale=False, shrink_tau=0.0),
    )

    # After first 4 windows, values should appear
    assert out3.first_valid_index() >= idx[4]
    assert out4.first_valid_index() >= idx[4]
    # Some finite values present
    assert np.isfinite(out3.dropna()).any()
    assert np.isfinite(out4.dropna()).any()


def test_panel_betas_shape_and_columns():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=25, freq="B")
    rm = pd.Series(rng.normal(size=len(idx)), index=idx)
    r = pd.DataFrame(
        {
            "A": 0.3 * (rm**2) + 0.1 * rng.normal(size=len(idx)),
            "B": -0.2 * (rm**3) + 0.1 * rng.normal(size=len(idx)),
        },
        index=idx,
    )

    out = panel_betas(
        r,
        rm,
        window=8,
        min_obs=4,
        which="coskew",
        cfg=CoMomentConfig(winsor=0.0, robust_scale=False, shrink_tau=0.0),
    )
    # aligns to r.index, columns preserved
    assert out.index.equals(r.index)
    assert list(out.columns) == ["A", "B"]
    # Some values computed after warmup
    assert out.iloc[10:].notna().any().any()


def test__demean_centers_series():
    s = pd.Series([1.0, 2.0, 3.0])
    d = _demean(s)
    assert pytest.approx(d.tolist()) == [-1.0, 0.0, 1.0]
    assert d.mean() == pytest.approx(0.0)


def test_joint_cumulant4_empty_returns_nan():
    a = pd.Series([np.nan, np.nan])
    b = pd.Series([np.nan, np.nan])
    c = pd.Series([np.nan, np.nan])
    d = pd.Series([np.nan, np.nan])
    assert np.isnan(joint_cumulant4(a, b, c, d, center=True))


def test_scale_invariance_with_robust_scale():
    rng = np.random.default_rng(1)
    n = 2000
    rm = pd.Series(rng.normal(size=n))
    ri = pd.Series(rng.normal(size=n))
    cfg = CoMomentConfig(robust_scale=True, winsor=0.0, shrink_tau=0.0)

    b3 = coskew_beta(ri, rm, cfg)
    b3_scaled = coskew_beta(ri, 10.0 * rm, cfg)  # rescale market
    b4 = cokurt_beta(ri, rm, cfg)
    b4_scaled = cokurt_beta(ri, 10.0 * rm, cfg)

    assert (
        np.isfinite(b3)
        and np.isfinite(b3_scaled)
        and np.isfinite(b4)
        and np.isfinite(b4_scaled)
    )
    assert abs(b3 - b3_scaled) < 1e-8
    assert abs(b4 - b4_scaled) < 1e-8
