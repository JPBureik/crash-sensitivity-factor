[![CI](https://github.com/JPBureik/crash-sensitivity-factor/actions/workflows/ci.yml/badge.svg)](https://github.com/JPBureik/crash-sensitivity-factor/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JPBureik/crash-sensitivity-factor/graph/badge.svg)](https://codecov.io/gh/JPBureik/crash-sensitivity-factor)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

# Crash-Sensitivity Factor (`crashload`)

A library for computing **crash-sensitivity signals** from higher-order co-moments between individual stocks and the market.

---

## Methodology

### The Problem

Standard beta (β) measures linear exposure to market returns, but says nothing about how a stock behaves during **extreme market moves**. Two stocks with identical betas can behave very differently in a crash.

### The Solution: Higher-Order Co-Moments

This library computes **coskewness** and **cokurtosis** betas that capture nonlinear crash exposure:

**Coskewness Beta (β³)** — measures asymmetric downside exposure:

```
β³ = κ(rᵢ, rₘ, rₘ) / Var(rₘ)^(3/2)
```

where κ(rᵢ, rₘ, rₘ) is the third joint cumulant. With centered returns: `κ = E[rᵢ · rₘ²]`.

- **β³ < 0**: Stock underperforms on large market moves (both up and down) → crash-sensitive
- **β³ > 0**: Stock outperforms on large market moves → crash-resilient

**Cokurtosis Beta (β⁴)** — measures tail-regime exposure:

```
β⁴ = κ(rᵢ, rₘ, rₘ, rₘ) / Var(rₘ)²
```

where κ is the fourth joint cumulant. With centered returns: `κ = E[rᵢ · rₘ³] − 3·E[rᵢ·rₘ]·E[rₘ²]`.

- **β⁴ > 0**: Stock amplifies extreme market moves → high tail sensitivity
- **β⁴ < 0**: Stock dampens extreme market moves → low tail sensitivity

### Crash Load Score (CLS)

The final signal combines both betas into a single cross-sectional score:

```
CLS = z(−β³) + λ · z(β⁴)
```

where z(·) is the cross-sectional robust z-score (median/MAD), and λ weights the kurtosis term (default 0.5).

**Interpretation**: Higher CLS → more crash-sensitive. Useful for:
- Risk monitoring (flag high-CLS positions)
- Factor construction (long low-CLS, short high-CLS)
- Regime analysis (track market-wide CLS distribution)

### Robust Estimation

Higher-order moments are notoriously noisy. This library applies:
- **Winsorization**: Clip extreme returns (default: 1% tails)
- **MAD scaling**: Normalize by median absolute deviation instead of std
- **Ridge shrinkage**: Shrink betas toward zero based on sample size

---

## Quick Start

```bash
git clone https://github.com/JPBureik/crash-sensitivity-factor.git
cd crash-sensitivity-factor
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Run the Example (no external data needed)

```bash
python examples/synthetic_demo.py
```

This generates synthetic returns with known crash profiles and computes CLS:

```
Computed CLS: 12 month-ends x 12 names
Last month-end top 8 (higher = more crash/tail-sensitive):
CRASHY_00    1.483
CRASHY_01    1.347
CRASHY_02    1.291
...
```

The demo creates three stock types:
- `NEUTRAL_*`: Linear market exposure only
- `CRASHY_*`: Negative coskewness (underperform on large moves)
- `TAIL_*`: Positive cokurtosis (amplify extreme moves)

### With Your Own Data

```python
import crashload as cl

# r: DataFrame of daily returns (rows=dates, cols=tickers)
# rm: Series of market returns (e.g., SPY), same index as r

cfg = cl.CoMomentConfig(
    winsor=0.01,        # clip 1% tails
    robust_scale=True,  # MAD normalization
    shrink_tau=100.0,   # ridge shrinkage strength
)

cls = cl.crash_score(
    r, rm,
    window=504,         # 2-year rolling window
    min_obs=252,        # require 1 year minimum
    lambda_=0.5,        # weight on β⁴ term
    cfg=cfg,
    monthly=True,       # resample to month-end
)
```

---

## API Reference

| Function | Description |
|----------|-------------|
| `coskew_beta(ri, rm, cfg)` | Single-stock β³ |
| `cokurt_beta(ri, rm, cfg)` | Single-stock β⁴ |
| `rolling_beta_series(ri, rm, window, ...)` | Rolling β³ or β⁴ for one stock |
| `panel_betas(r, rm, window, ...)` | Rolling betas for all stocks |
| `crash_score(r, rm, window, ...)` | Full CLS computation |
| `load_universe(path)` | Load S&P 500 universe CSV |

> Full docstrings in source: `python -c "import crashload; help(crashload.crash_score)"`

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

**Reproduce the universe snapshot:**
```bash
python scripts/get_sp500_tickers.py --out data/universe/sp500_$(date +%F).csv
```

---

## Development

```bash
# Run tests
pytest

# Run linters
ruff check src tests
black --check src tests
mypy
```

---

## License

MIT. See `LICENSE`.

*For research/education only. Not investment advice.*