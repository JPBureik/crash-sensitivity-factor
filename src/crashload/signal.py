from __future__ import annotations

import pandas as pd

from .estimators import CoMomentConfig, panel_betas
from .utils import robust_zscore


def crash_score(
    r: pd.DataFrame,
    rm: pd.Series,
    window: int = 504,
    min_obs: int = 252,
    lambda_: float = 0.5,
    cfg: CoMomentConfig | None = None,
    monthly: bool = True,
) -> pd.DataFrame:
    """
    Compute Crash Load Score (CLS) per name over time:
      CLS = z( -β_coskew ) + λ * z( β_cokurt )
    Returns a DataFrame aligned to r.index (or month-ends if monthly=True).
    """
    cfg = cfg or CoMomentConfig(winsor=0.0, robust_scale=True, shrink_tau=100.0)

    beta3 = panel_betas(r, rm, window, min_obs, which="coskew", cfg=cfg)
    beta4 = panel_betas(r, rm, window, min_obs, which="cokurt", cfg=cfg)

    # z-score cross-sectionally per date
    def z_xs(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(robust_zscore, axis=1)

    z3 = z_xs(-beta3)  # negative coskew = worse in crashes
    z4 = z_xs(beta4)
    cls = z3 + lambda_ * z4

    if monthly:
        # stamp to last available day of each month by taking last-observation
        cls = cls.resample("ME").last()

    return cls
