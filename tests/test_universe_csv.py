# tests/test_universe_csv.py
"""
Validate the frozen S&P 500 universe CSV produced by scripts/get_sp500_tickers.py.

Defaults:
- looks for the newest file matching data/universe/sp500_*.csv
- override with env var CRASHLOAD_UNIVERSE=/full/path/to/file.csv

Checks:
- required columns exist: ticker, name, sector
- row count in a reasonable range (400..550)
- no NaNs; names non-empty after strip
- tickers unique (case-insensitive) and well-formed (A-Z0-9 and single dashes)
- sectors in the canonical GICS 11 set (after whitespace normalization)
"""

from __future__ import annotations

import glob
import os
import pathlib
import re

import pandas as pd
import pytest

CANON_SECTORS = {
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
}

# Accept only uppercase letters, digits, and dashes.
# No leading/trailing dash, and no consecutive dashes.
TICKER_RE = re.compile(r"^(?!-)(?!.*--)[A-Z0-9-]+(?<!-)$")


@pytest.fixture(scope="session")
def universe_path() -> pathlib.Path:
    override = os.environ.get("CRASHLOAD_UNIVERSE")
    if override:
        p = pathlib.Path(override)
        if not p.exists():
            pytest.skip(f"CRASHLOAD_UNIVERSE set but file not found: {p}")
        return p

    paths = sorted(glob.glob("data/universe/sp500_*.csv"))
    if not paths:
        pytest.skip(
            "No universe CSV found. Run:\n"
            "  python scripts/get_sp500_tickers.py --out data/universe/sp500_$(date +%F).csv"
        )
    return pathlib.Path(paths[-1])  # newest by name


@pytest.fixture(scope="session")
def df_universe(universe_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(universe_path)
    # Normalize common issues early
    for col in ("ticker", "name", "sector"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def test_required_columns_present(df_universe: pd.DataFrame) -> None:
    required = {"ticker", "name", "sector"}
    missing = required - set(df_universe.columns)
    assert (
        not missing
    ), f"Missing required columns: {sorted(missing)}; got {list(df_universe.columns)}"


def test_row_count_reasonable(df_universe: pd.DataFrame) -> None:
    n = len(df_universe)
    assert (
        400 <= n <= 550
    ), f"Unexpected number of rows: {n} (expected ~500; multiple share classes can exceed 500)"


def test_no_nans_and_names_nonempty(df_universe: pd.DataFrame) -> None:
    # No NaNs in required columns
    assert (
        df_universe[["ticker", "name", "sector"]].notna().all().all()
    ), "Found NaNs in required columns"
    # Strip & check name not empty
    empties = df_universe["name"].str.strip().eq("")
    assert not empties.any(), f"{empties.sum()} company names are empty after stripping"


def test_tickers_unique_case_insensitive(df_universe: pd.DataFrame) -> None:
    # Clean whitespace/case
    tick = df_universe["ticker"].astype(str).str.strip()
    dups = tick.str.upper().duplicated(keep=False)
    assert (
        not dups.any()
    ), f"Found duplicate tickers (case-insensitive): {tick[dups].tolist()[:10]}..."


def test_ticker_format(df_universe: pd.DataFrame) -> None:
    tick = df_universe["ticker"].astype(str).str.strip()
    bad = tick[~tick.str.match(TICKER_RE)]
    msg = (
        "Invalid ticker format detected. "
        "Allowed: A-Z, 0-9, single dashes; no leading/trailing dash or double dashes. "
        f"Examples: {bad.head(10).tolist()}"
    )
    assert bad.empty, msg


def test_sector_values(df_universe: pd.DataFrame) -> None:
    # Normalize whitespace & capitalization (Wikipedia is usually consistent)
    sect = df_universe["sector"].astype(str).str.strip()
    # Keep original for error display
    invalid = sect[~sect.isin(CANON_SECTORS)]
    assert invalid.empty, (
        "Found sectors outside canonical GICS 11. "
        f"Invalid examples: {invalid.value_counts().head(10).to_dict()}. "
        f"Expected one of: {sorted(CANON_SECTORS)}"
    )
