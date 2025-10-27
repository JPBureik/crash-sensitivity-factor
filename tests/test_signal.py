# tests/test_signal.py
import numpy as np
import pandas as pd
import pytest

import crashload.signal as sig  # we’ll monkeypatch sig.panel_betas
from crashload.signal import crash_score, CoMomentConfig


def _make_index(n_days=20, start="2024-01-03"):
    # Business-day index spanning 2 months for the monthly resample test
    return pd.bdate_range(start, periods=n_days)


def test_crash_score_nonmonthly_z_logic(monkeypatch):
    idx = _make_index(10)
    r = pd.DataFrame(0.0, index=idx, columns=["A", "B", "C"])
    rm = pd.Series(0.0, index=idx)

    # Fake panel_betas to return known per-row values:
    #   coskew -> [1, 2, 3]  -> after negation becomes [-1, -2, -3]
    #   cokurt -> [5, 5, 5]  -> constant -> z = 0 row-wise
    def fake_panel_betas(r_, rm_, window, min_obs, which="coskew", cfg=None):
        base = pd.DataFrame(index=r_.index, columns=r_.columns, dtype=float)
        if which == "coskew":
            base.loc[:, "A"] = 1.0
            base.loc[:, "B"] = 2.0
            base.loc[:, "C"] = 3.0
        else:  # "cokurt"
            base.loc[:, :] = 5.0
        return base

    monkeypatch.setattr(sig, "panel_betas", fake_panel_betas)

    out = crash_score(r, rm, window=5, min_obs=3, lambda_=0.5, cfg=CoMomentConfig(), monthly=False)

    # Expected row-wise z of [-1, -2, -3]:
    scale = 1.4826  # MAD scaling used by robust_zscore
    z3_row = pd.Series([-1.0, -2.0, -3.0], index=["A", "B", "C"])
    z3_row = (z3_row - z3_row.median()) / scale  # -> [~0.6745, 0, ~-0.6745]
    # z4 row is all zeros (constant input) ⇒ CLS = z3_row
    first_non_na = out.iloc[0]
    assert first_non_na["A"] == pytest.approx(z3_row["A"], rel=1e-6, abs=1e-6)
    assert first_non_na["B"] == pytest.approx(z3_row["B"], rel=1e-6, abs=1e-6)
    assert first_non_na["C"] == pytest.approx(z3_row["C"], rel=1e-6, abs=1e-6)
    assert out.index.equals(r.index)
    assert list(out.columns) == ["A", "B", "C"]


def test_crash_score_monthly_resample_and_lambda_weight(monkeypatch):
    idx = _make_index(30)  # spans into the next month
    r = pd.DataFrame(0.0, index=idx, columns=["A", "B", "C"])
    rm = pd.Series(0.0, index=idx)

    # Now make both coskew and cokurt non-constant so lambda_ matters:
    #   coskew -> [1,2,3]  -> z([-1,-2,-3]) as before
    #   cokurt -> [1,2,3]  -> z([1,2,3]) = negative of z([-1,-2,-3])
    def fake_panel_betas(r_, rm_, window, min_obs, which="coskew", cfg=None):
        base = pd.DataFrame(index=r_.index, columns=r_.columns, dtype=float)
        vals = [1.0, 2.0, 3.0]
        for col, v in zip(base.columns, vals):
            base[col] = v
        return base

    monkeypatch.setattr(sig, "panel_betas", fake_panel_betas)

    out = crash_score(r, rm, window=5, min_obs=3, lambda_=0.5,
                      cfg=CoMomentConfig(), monthly=True)

    # Monthly resample yields month-end stamps; ensure index is month-end dates
    assert out.index.to_series().dt.is_month_end.all()
    # Expected CLS row per construction:
    scale = 1.4826
    z3 = (pd.Series([-1.0, -2.0, -3.0], index=["A", "B", "C"]) -
          (-2.0)) / scale
    z4 = (pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"]) -
          (2.0)) / scale
    expected = z3 + 0.5 * z4  # -> approx [0.33725, 0, -0.33725]

    last_row = out.iloc[-1]
    assert last_row["A"] == pytest.approx(expected["A"], rel=1e-6, abs=1e-6)
    assert last_row["B"] == pytest.approx(expected["B"], rel=1e-6, abs=1e-6)
    assert last_row["C"] == pytest.approx(expected["C"], rel=1e-6, abs=1e-6)
