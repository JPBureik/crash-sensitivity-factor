# Data guide

This project is designed to be **reproducible and legally clean**: no bulk data is stored in the repo; users fetch locally with provided scripts. Outputs are deterministic given a **frozen universe** and the same fetch date.

---

## 1) Universe snapshot (v1.0)

- **Source:** Wikipedia “List of S&P 500 companies” scraped on the day you run the script.  
- **File:** `data/universe/sp500_YYYY-MM-DD.csv`  
- **Columns (required):**
  - `ticker` — upper-case, dots converted to dashes (e.g., `BRK-B`), deduplicated.
  - `name` — company name (string).
  - `sector` — GICS sector (string). *(Sub-industry optional but helpful.)*
- **Frozen-universe note:** We **do not** backfill future entrants or remove future deletions; we keep the exact list from this date. Names without sufficient history simply have missing early data.
- **Recommendation:** **Commit this CSV** so analyses are tied to a specific “as-of” date.

**Command**
```bash
python scripts/get_sp500_tickers.py --out data/universe/sp500_$(date +%F).csv
```

---

## 2) Price & return series

We compute **daily total-return proxies** per ticker using adjusted close.

- **Input source:** publicly accessible APIs (e.g., Yahoo via yfinance or Stooq).  
- **Raw file (optional):** `data/raw/ohlcv_<SOURCE>_<YYYY-MM-DD>.parquet`  
  - Columns: `date` (YYYY-MM-DD), `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `source`.
- **Returns file (required):** `data/returns/returns.parquet` (**not committed**)  
  - Columns:
    - `date` — trading date at U.S. close (no timezone).
    - `ticker`
    - `ret` — daily **log return** from **adjusted close** (corporate actions accounted for).
  - Invariants / checks:
    - One row per `(date, ticker)`, no duplicates.
    - No infinite or NaN `ret` (after filtering).
    - Dates strictly increasing for each ticker.

**Commands**
```bash
# Fetch raw prices for the frozen universe window
python scripts/get_prices.py   --universe data/universe/sp500_YYYY-MM-DD.csv   --start 2005-01-01 --end 2025-10-31 --out data/raw/

# Build daily returns (light winsorization to catch bad prints)
python scripts/make_returns.py   --raw data/raw/ --out data/returns/returns.parquet
```

**Data hygiene in `make_returns.py`:**
- Drop days with zero or missing `adj_close`.
- Compute `ret = ln(adj_close_t / adj_close_{t-1})`.
- Optional light **winsorization** at the 0.01% tails to catch obvious bad prints.
- Keep only tickers in the **frozen** universe file.

---

## 3) Market series (benchmark for co-moments)

You need one **market return** series aligned to your daily panel.

- **Preferred:** SPY adjusted-close returns as a total-return proxy.  
- **Alternative:** S&P 500 index (`^GSPC`) returns (price index; lacks dividends, but acceptable for v0.1).  
- **File:** `data/market/market_returns.parquet`  
  - Columns: `date`, `ret_mkt`.
- **Alignment:** same trading calendar as equities; inner-join on `date`.

**Command**
```bash
python scripts/make_market_series.py --start 2005-01-01 --end 2025-10-31   --symbol SPY --out data/market/market_returns.parquet
```

*(Optional)* add `data/riskfree/rf.parquet` with `date, rf` if you want excess returns.

---

## 4) Calendars & stamping

- **Trading calendar:** U.S. market close dates. Non-trading days are excluded.  
- **Time zone:** dates are **close-of-day** (no timezone stored).  
- **Monthly stamping:** signals are computed on the **last available trading day of each month** using only data **up to and including** that day (no look-ahead).

**Generated file**
- `data/calendar/month_ends.parquet` with `month_end` trading dates.

---

## 5) Minimum data requirements (per ticker)

- At least **500 trading days** of history before a month-end to compute the signal.
- Continuous stretch check: last missing gap ≤ 5 days near month-end (otherwise skip that month-end for that ticker).

---

## 6) Industry data (for neutralization)

- **Source:** from the universe snapshot (sector), or a separate mapping.  
- **File:** `data/metadata/industry.csv`  
  - Columns: `ticker`, `sector`[, `sub_industry`].
- Used to **neutralize** sector effects in cross-sectional ranking.

---

## 7) Directory layout summary

```text
data/
  universe/      sp500_YYYY-MM-DD.csv           # committed (frozen list)
  raw/           ohlcv_SOURCE_*.parquet         # not committed
  returns/       returns.parquet                # not committed
  market/        market_returns.parquet         # not committed
  riskfree/      rf.parquet                     # optional, not committed
  metadata/      industry.csv                   # committed
  calendar/      month_ends.parquet             # generated
```

---

## 8) Reproduce v0.1 data in one go

```bash
# 1) Universe
python scripts/get_sp500_tickers.py --out data/universe/sp500_YYYY-MM-DD.csv

# 2) Prices -> Returns
python scripts/get_prices.py --universe data/universe/sp500_YYYY-MM-DD.csv   --start 2005-01-01 --end 2025-10-31 --out data/raw/
python scripts/make_returns.py --raw data/raw/ --out data/returns/returns.parquet

# 3) Market
python scripts/make_market_series.py --symbol SPY --start 2005-01-01 --end 2025-10-31   --out data/market/market_returns.parquet
```

---

## 9) Known pitfalls

- **Survivorship bias:** freezing the universe **today** means earlier history includes only firms that survived to the snapshot date; this is intentional for v0.1 but should be called out in the report.
- **Corporate actions:** rely on **adjusted close**; if your source mis-adjusts a split/dividend, you’ll see a giant outlier—winsorization helps, but inspect.
- **Ticker mapping:** tickers can change; since we freeze today’s tickers, earlier periods may have **missing** data for names that didn’t exist—this is expected.
- **Index vs ETF:** SPY is a convenient total-return proxy; `^GSPC` omits dividends (slightly different drift). Be consistent within a run.
