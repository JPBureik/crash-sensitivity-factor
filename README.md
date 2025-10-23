# Crash-Sensitivity Factor (`crashload`)
**Robust connected co-moment (3rd/4th-order) crash-sensitivity signal for S&P 500 equities.**

This repository implements a **crash-sensitivity factor** designed to quantify each stock’s tendency to suffer **disproportionate losses during market stress**.  
The methodology uses **connected co-moments** (joint cumulants) — specifically **coskewness** (3rd order) and **cokurtosis** (4th order) with the market — with attention to **robust estimation and reproducible research practices** (frozen universe, transaction costs, out-of-sample validation).

---

## What is this?
A compact, reproducible library and reference pipeline to compute a **Crash Load Score (CLS)** that combines:
- **Coskewness beta** (3rd order): does the stock underperform disproportionately on large **down** market days?
- **Cokurtosis beta** (4th order): is the stock exposed to **tail-heavy** or **volatile market regimes** beyond normal conditions?

It includes robust estimators (winsorization, k-statistics, shrinkage, optional downside conditioning) and provides utilities to rank, neutralize, and backtest the signal on a **frozen S&P 500** universe.

---

## Why it exists
Higher-order moments are powerful but notoriously noisy. This project aims to show how a fragile statistical concept can be turned into a **stable, interpretable, and risk-aware factor** through reproducible code, careful preprocessing, and transparent validation.

---

## Quick start (local dev)
```bash
# Clone:
git clone https://github.com/JPBureik/crash-sensitivity-factor.git
cd crash-sensitivity-factor

# (Optional) Create a virtual environment:
python -m venv .venv && source .venv/bin/activate

# Install in editable mode:
pip install -e .
```

---

## Minimal usage
```python
import pandas as pd
import crashload as cl  # alias 'cl' for convenience

# r: DataFrame of daily returns (rows: dates, cols: tickers)
# m: Series of daily market returns aligned to r.index
# Both should be de-duplicated, in UTC-close convention, etc.

score = cl.crash_score(r, m, lambda_=0.5, method="robust")
# Score: DataFrame indexed by date (e.g., month-end) with per-ticker CLS

# Example: build cross-sectional deciles at month-end:
dec = cl.rank_to_deciles(score)
ls_pnl = cl.long_short(dec, long_bucket=1, short_bucket=10, costs_bps=4)
```
Data is not provided. Place your inputs under `data/` (gitignored). See `data/README.md` for expected formats.

---

## Disclaimer
For research/education. Not investment advice.

---

## License
MIT. See `LICENSE`.