[![CI](https://github.com/JPBureik/crash-sensitivity-factor/actions/workflows/ci.yml/badge.svg)](https://github.com/JPBureik/crash-sensitivity-factor/actions/workflows/ci.yml)

# Crash-Sensitivity Factor (`crashload`)
**Robust connected co-moment (3rd/4th-order) crash-sensitivity signal for S&P 500 equities.**

This repository implements a **crash-sensitivity factor** designed to quantify each stock’s tendency to suffer **disproportionate losses during market stress**.  
The methodology uses **connected co-moments** (joint cumulants) — specifically **coskewness** (3rd order) and **cokurtosis** (4th order) with the market — with attention to **robust estimation and reproducible research practices** (frozen universe).

---

## What is this?
A compact, reproducible library (with an example workflow) to compute a **Crash Load Score (CLS)** that combines:
- **Coskewness beta** (3rd order): does the stock underperform disproportionately on large **down** market days?
- **Cokurtosis beta** (4th order): is the stock exposed to **tail-heavy** or **volatile market regimes** beyond normal conditions?

It includes robust preprocessing (winsorization, MAD scaling) and shrinkage, plus a tested API to compute rolling coskew/cokurt betas and a monthly CLS.

---

## API Overview

- `coskew_beta`, `cokurt_beta`
- `rolling_beta_series`, `panel_betas`
- `crash_score` (CLS construction: `z(-β3) + λ z(β4)`)

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
import crashload as cl

# r: DataFrame of daily returns (rows: dates, cols: tickers)
# rm: Series of daily market returns aligned to r.index
# Both should be de-duplicated, in UTC-close convention, etc.

cfg = cl.CoMomentConfig(winsor=0.01, robust_scale=True, shrink_tau=100.0)

cls = cl.crash_score(
    r, rm,
    window=504, min_obs=252,
    lambda_=0.5,
    cfg=cfg,
    monthly=True,   # month-end stamps; set False for daily
)
```
Data is not provided. Place your inputs under `data/` (gitignored). See `data/README.md` for expected formats.

---

## Data & Reproducibility

**Universe.** S&P 500 **snapshot** (fixed membership) as of 2025-10-27, stored at `data/universe/sp500_2025-10-27.csv`.

**Inputs.** The library expects user-supplied daily stock returns (`r`) and a market return series (`rm`). Data is not redistributed.

**Span (typical).** 2004-01-01 → present (includes GFC, 2011, 2015–16, 2018, 2020, 2022).

**Reproducibility.** Deterministic computations + test suite. Includes a script to reproduce the S&P 500 snapshot; ingestion of prices/returns is intentionally left to the user.

**Fair use.** This repository **does not redistribute** data. It only provides a script to reproduce the S&P 500 snapshot; users source prices/returns locally. Returns/market series are user-supplied (not redistributed).

**Known limitation (v1.0).** Static membership ⇒ **survivorship bias** (delisted names omitted). The main output — a *market-level* crash-sensitivity or regime signal — is less affected by survivorship, but any cross-sectional backtests are labeled **illustrative**.  

**Planned improvements.**  
- **v1.1:** Monthly ETF holdings (SPY/IVV/VOO) as a rolling universe.  
- **v2.0:** WRDS/CRSP constituents with delisted returns.

**Reproduce the data locally**
```bash
python scripts/get_sp500_tickers.py
```
 
 ---

 ## Validation

Tests cover statistical sanity checks (Gaussian ≈ 0), shape/rolling invariants, and CLS monthly stamping.
Reproduce the test results via:
```bash
pytest -q
```

## Runnable example (no external data)

```bash
python examples/synthetic_demo.py
```

This generates synthetic daily returns, computes CLS at month-end, prints a small summary, and writes outputs to `examples/output/`.

 ---

 ## Roadmap

- Speed up panel_betas (vectorization/numba)
- Add CLI example runner that reads parquet and writes CLS outputs
- Add optional price ingestion helper

 ---

## Disclaimer
For research/education. Not investment advice.

---

## License
MIT. See `LICENSE`.