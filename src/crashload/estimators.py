from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import (
    ridge_shrink,
    robust_scale_mad,
    winsorize_series,
)

# ----------------------------
# Low-level joint cumulants
# ----------------------------


def _demean(a: pd.Series) -> pd.Series:
    return a - a.mean()


def joint_cumulant3(
    x: pd.Series, y: pd.Series, z: pd.Series, center: bool = True
) -> float:
    """
    κ(X,Y,Z) = E[XYZ] - E[X]E[YZ] - E[Y]E[XZ] - E[Z]E[XY] + 2 E[X]E[Y]E[Z].
    If center=True, uses centered variables so κ simplifies to E[x*y*z].
    """
    df = pd.concat([x, y, z], axis=1).dropna()
    if df.empty:
        return np.nan
    X, Y, Z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    if center:
        X, Y, Z = X - X.mean(), Y - Y.mean(), Z - Z.mean()
        return float((X * Y * Z).mean())
    # general formula
    EX, EY, EZ = X.mean(), Y.mean(), Z.mean()
    EXY, EXZ, EYZ = (X * Y).mean(), (X * Z).mean(), (Y * Z).mean()
    EXYZ = (X * Y * Z).mean()
    return float(EXYZ - EX * EYZ - EY * EXZ - EZ * EXY + 2 * EX * EY * EZ)


def joint_cumulant4(
    a: pd.Series, b: pd.Series, c: pd.Series, d: pd.Series, center: bool = True
) -> float:
    """
    κ4(a,b,c,d) = E[abcd]
      - E[ab]E[cd] - E[ac]E[bd] - E[ad]E[bc]
      + 2*E[a]E[b]E[cd] + 2*E[a]E[c]E[bd] + 2*E[a]E[d]E[bc]
      + 2*E[b]E[c]E[ad] + 2*E[b]E[d]E[ac] + 2*E[c]E[d]E[ab]
      - 6*E[a]E[b]E[c]E[d].
    If center=True, uses centered variables and reduces to
      κ4 = E[abcd] - E[ab]E[cd] - E[ac]E[bd] - E[ad]E[bc].
    """
    df = pd.concat([a, b, c, d], axis=1).dropna()
    if df.empty:
        return np.nan
    A, B, C, D = (df.iloc[:, i] for i in range(4))

    def E(s: pd.Series) -> float:
        return float(s.mean())

    if center:
        A, B, C, D = (u - u.mean() for u in (A, B, C, D))
        term = E(A * B * C * D)
        term -= E(A * B) * E(C * D)
        term -= E(A * C) * E(B * D)
        term -= E(A * D) * E(B * C)
        return term
    # full formula
    term = E(A * B * C * D)
    term -= E(A * B) * E(C * D)
    term -= E(A * C) * E(B * D)
    term -= E(A * D) * E(B * C)
    term += 2 * (
        E(A) * E(B) * E(C * D) + E(A) * E(C) * E(B * D) + E(A) * E(D) * E(B * C)
    )
    term += 2 * (
        E(B) * E(C) * E(A * D) + E(B) * E(D) * E(A * C) + E(C) * E(D) * E(A * B)
    )
    term -= 6 * E(A) * E(B) * E(C) * E(D)
    return term


# --------------------------------------
# Market-connected coskew/cokurt betas
# --------------------------------------


@dataclass(frozen=True)
class CoMomentConfig:
    winsor: float = 0.0  # e.g., 0.01 for 1% tails
    robust_scale: bool = True  # scale series by MAD
    shrink_tau: float = 0.0  # ridge shrink n/(n+tau)
    use_centered: bool = True  # center before cumulants


def _prep_pair(
    ri: pd.Series, rm: pd.Series, cfg: CoMomentConfig
) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([ri, rm], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    x, m = df.iloc[:, 0].copy(), df.iloc[:, 1].copy()

    # Optional winsorization (per-series)
    if cfg.winsor and cfg.winsor > 0:
        x = winsorize_series(x, cfg.winsor, 1 - cfg.winsor)
        m = winsorize_series(m, cfg.winsor, 1 - cfg.winsor)

    # Center
    if cfg.use_centered:
        x = x - x.mean()
        m = m - m.mean()

    # Optional robust scaling for numerical stability (does not change betas)
    if cfg.robust_scale:
        sx, sm = robust_scale_mad(x), robust_scale_mad(m)
        if sx > 0 and np.isfinite(sx):
            x = x / sx
        if sm > 0 and np.isfinite(sm):
            m = m / sm

    return x, m


def coskew_beta(
    ri: pd.Series, rm: pd.Series, cfg: CoMomentConfig | None = None
) -> float:
    """
    β^(3) = κ(ri, rm, rm) / Var(rm)^(3/2).
    With centered variables, κ(ri, rm, rm) = E[ri * rm^2].
    """
    cfg = cfg or CoMomentConfig()
    x, m = _prep_pair(ri, rm, cfg)
    if len(x) == 0:
        return np.nan
    num = joint_cumulant3(x, m, m, center=True)
    den = float(m.var()) ** 1.5
    if den <= 0 or not np.isfinite(den):
        return np.nan
    val = num / den
    if cfg.shrink_tau and cfg.shrink_tau > 0:
        val = ridge_shrink(val, n_eff=len(x), tau=cfg.shrink_tau, target=0.0)
    return float(val)


def cokurt_beta(
    ri: pd.Series, rm: pd.Series, cfg: CoMomentConfig | None = None
) -> float:
    """
    β^(4) = κ(ri, rm, rm, rm) / Var(rm)^2.
    With centered variables, κ = E[ri * rm^3] - 3 E[ri*rm] E[rm^2].
    """
    cfg = cfg or CoMomentConfig()
    x, m = _prep_pair(ri, rm, cfg)
    if len(x) == 0:
        return np.nan
    # use centered simplification
    num = joint_cumulant4(x, m, m, m, center=True)  # equals E[x*m^3] - 3 E[x*m] E[m^2]
    den = float(m.var()) ** 2
    if den <= 0 or not np.isfinite(den):
        return np.nan
    val = num / den
    if cfg.shrink_tau and cfg.shrink_tau > 0:
        val = ridge_shrink(val, n_eff=len(x), tau=cfg.shrink_tau, target=0.0)
    return float(val)


# --------------------------------------
# Rolling / panel helpers
# --------------------------------------


def rolling_beta_series(
    ri: pd.Series,
    rm: pd.Series,
    window: int = 504,
    min_obs: int = 252,
    which: str = "coskew",
    cfg: CoMomentConfig | None = None,
) -> pd.Series:
    """
    Rolling coskew/cokurt beta for a single asset vs market.
    Returns a Series aligned to ri index with NaNs until min_obs.
    """
    cfg = cfg or CoMomentConfig()
    ri, rm = ri.astype(float), rm.astype(float)
    joined = pd.concat([ri, rm], axis=1, keys=["ri", "rm"]).dropna()
    if joined.empty:
        return pd.Series(index=ri.index, dtype=float)

    func = coskew_beta if which == "coskew" else cokurt_beta
    out = []
    idx = []
    vals = joined.index
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        sl = joined.iloc[start : i + 1]
        if len(sl) >= min_obs:
            out.append(func(sl["ri"], sl["rm"], cfg))
            idx.append(vals[i])
    return pd.Series(out, index=idx).reindex(ri.index)


def panel_betas(
    r: pd.DataFrame,
    rm: pd.Series,
    window: int = 504,
    min_obs: int = 252,
    which: str = "coskew",
    cfg: CoMomentConfig | None = None,
) -> pd.DataFrame:
    """
    Compute rolling betas for each column in r against rm.
    Returns a DataFrame aligned to r.index, columns=r.columns.
    """
    cfg = cfg or CoMomentConfig()
    out = {}
    for col in r.columns:
        out[col] = rolling_beta_series(
            r[col], rm, window, min_obs, which=which, cfg=cfg
        )
    return pd.DataFrame(out, index=r.index)
