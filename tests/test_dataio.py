# tests/test_dataio.py
import pandas as pd
import pytest

from crashload.dataio import load_universe


def test_load_universe_with_explicit_path_trims(tmp_path):
    p = tmp_path / "sp500_sample.csv"
    df = pd.DataFrame(
        {
            "ticker": [" aapl  ", "MSFT "],
            "name": [" Apple Inc. ", " Microsoft Corp "],
            "sector": [" Information Technology ", "Information Technology"],
        }
    )
    df.to_csv(p, index=False)

    out = load_universe(str(p))
    assert out.loc[0, "ticker"] == "aapl"
    assert out.loc[0, "name"] == "Apple Inc."
    assert out.loc[0, "sector"] == "Information Technology"
    # nothing weird happened to other cols
    assert set(["ticker", "name", "sector"]).issubset(out.columns)


def test_load_universe_autodiscover_picks_newest(monkeypatch, tmp_path):
    # work in an isolated cwd so the hardcoded glob path is under tmp_path
    monkeypatch.chdir(tmp_path)
    unidir = tmp_path / "data" / "universe"
    unidir.mkdir(parents=True)

    old = pd.DataFrame({"ticker": ["OLD"], "name": ["Old Co"], "sector": ["Energy"]})
    new = pd.DataFrame({"ticker": ["NEW"], "name": ["New Co"], "sector": ["Utilities"]})

    old.to_csv(unidir / "sp500_2024-12-31.csv", index=False)
    new.to_csv(unidir / "sp500_2025-01-01.csv", index=False)

    out = load_universe()  # should pick 2025-01-01 by sorted filename
    assert out.iloc[0]["ticker"] == "NEW"
    assert len(out) == 1


def test_load_universe_no_files_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no data/universe created → glob finds nothing
    with pytest.raises(FileNotFoundError):
        load_universe()
