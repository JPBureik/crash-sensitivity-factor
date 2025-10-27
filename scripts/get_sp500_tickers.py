#!/usr/bin/env python
# Get S&P 500 tickers from Wikipedia, print an overview, and optionally save a CSV.

import argparse
import pathlib
import sys
from io import StringIO

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

def p(msg: str) -> None:
    print(msg, flush=True)

def clean_ticker(sym: str) -> str:
    s = str(sym).strip().upper().replace(".", "-")
    return s

def _norm_cols(cols) -> list[str]:
    """Flatten possible MultiIndex columns and normalize to lowercase strings."""
    def flatten_one(col):
        if isinstance(col, tuple):
            parts = [
                str(x).strip()
                for x in col
                if x is not None and str(x).strip() and not str(x).startswith("Unnamed")
            ]
            if not parts:
                parts = [str(col[-1]) if col and col[-1] is not None else ""]
            name = " ".join(parts)
        else:
            name = str(col)
        return name.replace("\xa0", " ").strip().lower()
    return [flatten_one(c) for c in list(cols)]

def pick_constituents_table(html: str) -> pd.DataFrame:
    """Return the table that looks like the S&P 500 constituents."""
    p("Parsing HTML tables from Wikipedia…")
    # StringIO fixes the FutureWarning about literal HTML
    all_tables = pd.read_html(StringIO(html), header=0)
    p(f"Found {len(all_tables)} tables.")

    candidates = []
    for i, t in enumerate(all_tables):
        cols = _norm_cols(t.columns)
        has_symbol = any(c in ("symbol", "ticker") for c in cols)
        has_name = any(c in ("security", "company", "name") for c in cols)
        has_sector = any(c == "gics sector" for c in cols)
        if has_symbol and has_name and has_sector:
            tt = t.copy()
            tt.columns = _norm_cols(tt.columns)
            candidates.append((i, tt, cols, tt.shape))

    if not candidates:
        raise RuntimeError(
            "Could not find an S&P 500-like table (need columns incl. Symbol/Security/GICS Sector)."
        )

    # Prefer the largest candidate (the constituents table is usually the largest)
    idx, table, cols, shape = max(candidates, key=lambda x: x[3][0])
    p(f"Chose table #{idx} with shape {shape} and columns: {cols[:6]}{' …' if len(cols) > 6 else ''}")
    return table

def tidy(df_raw: pd.DataFrame) -> pd.DataFrame:
    # df_raw already has normalized lowercase columns
    rename_map = {}
    for src, dst in (
        ("symbol", "ticker"), ("ticker", "ticker"),
        ("security", "name"), ("company", "name"), ("name", "name"),
        ("gics sector", "sector")
    ):
        if src in df_raw.columns:
            rename_map[src] = dst

    df = df_raw.rename(columns=rename_map)
    keep = [c for c in ("ticker", "name", "sector") if c in df.columns]
    if "ticker" not in df.columns:
        raise RuntimeError(f"Missing ticker-like column. Columns seen: {list(df_raw.columns)}")

    df = df[keep].copy()
    df["ticker"] = df["ticker"].map(clean_ticker)
    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df

def print_overview(df: pd.DataFrame) -> None:
    p("\n=== S&P 500 scrape overview ===")
    p(f"Rows: {len(df)} | Columns: {list(df.columns)}")
    cols = df.columns[:3]
    p("\nSample (first 10 rows of first 3 columns):")
    p(df.loc[:9, cols].to_string(index=False))

    if "sector" in df.columns:
        p("\nSector breakdown (top 10):")
        p(df["sector"].value_counts().head(10).to_string())

    dups = df["ticker"].duplicated().sum()
    if dups:
        p(f"\nWARNING: Found {dups} duplicate tickers after cleaning.")

    if len(df) > 500:
        p("\nNote: >500 can be normal—some companies have multiple share classes (e.g., GOOG/GOOGL).")

def main():
    # Ensure line-buffered prints on recent Python versions
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Fetch S&P 500 constituents from Wikipedia")
    ap.add_argument("--out", help="Output CSV path (e.g., data/universe/sp500_YYYY-MM-DD.csv)")
    ap.add_argument("--yes", action="store_true", help="Save without prompting (if --out is provided)")
    args = ap.parse_args()

    p(f"Fetching {WIKI_URL} …")
    try:
        resp = requests.get(WIKI_URL, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        p(f"ERROR: failed to fetch Wikipedia: {e}")
        sys.exit(1)

    df_raw = pick_constituents_table(resp.text)
    p("Tidying table…")
    df = tidy(df_raw)
    print_overview(df)

    if args.out:
        save = args.yes
        if not save:
            ans = input(f"\nSave {len(df)} tickers to {args.out}? [y/N]: ").strip().lower()
            save = ans in ("y", "yes")
        if save:
            out_path = pathlib.Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            p(f"Wrote {len(df)} tickers to {out_path}")
        else:
            p("Skipping save (no changes written).")
    else:
        p("\nNo --out provided. Showing overview only (nothing saved).")

if __name__ == "__main__":
    main()
