# Data guide

This project is designed to be **reproducible and legally clean**: no bulk data is stored in the repo; users fetch locally with their preferred provider; this repo only includes a script to reproduce the frozen universe snapshot. Outputs are deterministic given the same inputs.

---

## 1) Universe snapshot (v1.0)

- **Source:** Wikipedia "List of S&P 500 companies" scraped on the day you run the script.
- **File pattern:** `data/universe/sp500_YYYY-MM-DD.csv`
- **This repo includes:** `data/universe/sp500_2025-10-27.csv` (frozen snapshot used for v0.1)
- **Columns (required):**
  - `ticker` — upper-case, dots converted to dashes (e.g., `BRK-B`), deduplicated
  - `name` — company name (string)
  - `sector` — GICS sector (string). *(Sub-industry optional but helpful.)*
- **Frozen-universe note:** We **do not** backfill future entrants or remove future deletions; we keep the exact list from this date. Names without sufficient history simply have missing early data.

### Snapshot policy
For reproducibility, this repo keeps **one frozen snapshot**: `data/universe/sp500_2025-10-27.csv` (~500 rows).
If you remove it or want a fresh snapshot, regenerate with:
```bash
python scripts/get_sp500_tickers.py --out data/universe/sp500_$(date +%F).csv
```

### Attribution
This snapshot is derived from Wikipedia’s "List of S&P 500 companies" and used under **CC BY-SA 4.0**.
See `data/universe/ATTRIBUTION.md` for details.

---

## 2) Inputs expected by the library (not provided)

You must supply (1) daily stock returns `r` (dates × tickers) and (2) a daily market return series `rm` aligned on the same trading calendar. This repo intentionally does not redistribute price/return data, and does not yet ship ingestion scripts.

---

## 3) Planned tooling

- Calendar/month-ends parquet generation
- Sector neutralization / `data/metadata/industry.csv`
- Minimum data requirements rules
- Industry data (for neutralization)
- Directory layout that includes metadata/calendar files
- Implement new script to reproduce v0.1 data in one go:
  - `scripts/get_prices.py`
  - `scripts/make_returns.py`
  - `scripts/make_market_series.py`

---

## 4) Known pitfalls

- **Survivorship bias:** freezing the universe today means earlier history includes only firms that survived to the snapshot date; this is intentional for v0.1 but should be called out in the report
- **Corporate actions:** rely on **adjusted close**; if your source mis-adjusts a split/dividend, you’ll see a giant outlier—winsorization helps, but inspect
- **Ticker mapping:** tickers can change; since we freeze today’s tickers, earlier periods may have **missing** data for names that didn’t exist—this is expected
- **Index vs ETF:** SPY is a convenient total-return proxy; `^GSPC` omits dividends (slightly different drift). Be consistent within a run
