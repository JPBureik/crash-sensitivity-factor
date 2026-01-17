#!/usr/bin/env python3
"""
Synthetic end-to-end demo for crashload.

Generates:
  - rm: market return Series (business-day index)
  - r:  panel of synthetic stock returns (DataFrame)

Then computes:
  - CLS = z(-beta3) + lambda * z(beta4), stamped to month-end.

No external data required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import crashload as cl


def make_synthetic_data(
    n_days: int,
    n_neutral: int,
    n_crashy: int,
    n_tail: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create synthetic daily returns with a few 'types':
      - neutral: mostly linear exposure to the market
      - crashy: negative coskew exposure (tends to lose on large |market| moves)
      - tail:   positive cokurt exposure (sensitive to tail regimes)
    """
    rng = np.random.default_rng(seed)

    idx = pd.bdate_range("2020-01-01", periods=n_days)

    # Market returns (rough daily scale)
    rm = pd.Series(0.01 * rng.normal(size=n_days), index=idx, name="rm")

    # Common components
    mu = 0.0002
    beta_mkt = 0.2
    eps = 0.01 * rng.normal(size=(n_days, n_neutral + n_crashy + n_tail))

    # Centered quadratic term: (rm^2 - E[rm^2])
    m2 = rm**2
    m2c = m2 - float(m2.mean())

    # Cubic term for tail dependence
    m3 = rm**3

    cols: list[str] = []
    series_list: list[pd.Series] = []
    j = 0

    # Neutral names
    for i in range(n_neutral):
        ri = mu + beta_mkt * rm + eps[:, j]
        cols.append(f"NEUTRAL_{i:02d}")
        series_list.append(pd.Series(ri, index=idx))
        j += 1

    # Crashy names: negative loading on centered rm^2 -> negative coskew beta
    gamma = 2.0
    for i in range(n_crashy):
        ri = mu + beta_mkt * rm - gamma * m2c + eps[:, j]
        cols.append(f"CRASHY_{i:02d}")
        series_list.append(pd.Series(ri, index=idx))
        j += 1

    # Tail names: positive loading on rm^3 -> positive cokurt beta
    delta = 20.0
    for i in range(n_tail):
        ri = mu + beta_mkt * rm + delta * m3 + eps[:, j]
        cols.append(f"TAIL_{i:02d}")
        series_list.append(pd.Series(ri, index=idx))
        j += 1

    r = pd.concat(series_list, axis=1)
    r.columns = cols
    r.index.name = "date"

    return r.astype(float), rm.astype(float)


def write_outputs(cls: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    # Always write CSV (no optional deps)
    cls.to_csv(out)

    # Best-effort parquet (optional dependency)
    parquet_path = out.with_suffix(".parquet")
    try:
        cls.to_parquet(parquet_path)
    except Exception:
        # pyarrow/fastparquet may not be installed; CSV is the guaranteed artifact.
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="Run crashload on synthetic data (no external inputs).")
    p.add_argument("--n-days", type=int, default=320, help="Number of business days to simulate.")
    p.add_argument("--n-neutral", type=int, default=6, help="Number of neutral names.")
    p.add_argument("--n-crashy", type=int, default=3, help="Number of crashy names (negative coskew).")
    p.add_argument("--n-tail", type=int, default=3, help="Number of tail names (positive cokurt).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for determinism.")
    p.add_argument("--window", type=int, default=126, help="Rolling window length (days).")
    p.add_argument("--min-obs", type=int, default=63, help="Minimum observations required to compute a beta.")
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.5, help="Weight on z(beta4).")
    p.add_argument("--winsor", type=float, default=0.01, help="Winsorization tail fraction (0 disables).")
    p.add_argument("--shrink-tau", type=float, default=100.0, help="Ridge shrinkage strength (0 disables).")
    p.add_argument("--no-robust-scale", action="store_true", help="Disable MAD scaling.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/synthetic_cls.csv"),
        help="Output path for CLS (CSV).",
    )
    args = p.parse_args()

    r, rm = make_synthetic_data(
        n_days=args.n_days,
        n_neutral=args.n_neutral,
        n_crashy=args.n_crashy,
        n_tail=args.n_tail,
        seed=args.seed,
    )

    cfg = cl.CoMomentConfig(
        winsor=float(args.winsor),
        robust_scale=not args.no_robust_scale,
        shrink_tau=float(args.shrink_tau),
    )

    cls = cl.crash_score(
        r,
        rm,
        window=int(args.window),
        min_obs=int(args.min_obs),
        lambda_=float(args.lambda_),
        cfg=cfg,
        monthly=True,
    )

    print(f"Computed CLS: {cls.shape[0]} month-ends x {cls.shape[1]} names")
    print("Last month-end top 8 (higher = more crash/tail-sensitive):")
    last = cls.dropna(how="all").iloc[-1].sort_values(ascending=False)
    print(last.head(8).to_string())

    write_outputs(cls, args.out)
    print(f"Wrote: {args.out} (and parquet if available)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
