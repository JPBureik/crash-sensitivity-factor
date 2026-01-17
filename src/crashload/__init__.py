"""
Top-level package API.

This module re-exports the public, stable surface area of the library while
keeping package import time fast via lazy imports.
"""

from __future__ import annotations

from importlib import import_module
from importlib import metadata as _metadata
from typing import TYPE_CHECKING, Any, Dict, Tuple

# Public API (kept alphabetical)
__all__ = (
    "CoMomentConfig",
    "cokurt_beta",
    "coskew_beta",
    "crash_score",
    "load_universe",
    "panel_betas",
    "rolling_beta_series",
)

# Optional, but useful for users and debugging
try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# For static type checkers / IDEs (no runtime import cost)
if TYPE_CHECKING:
    from .dataio import load_universe
    from .estimators import (CoMomentConfig, cokurt_beta, coskew_beta,
                             panel_betas, rolling_beta_series)
    from .signal import crash_score

# Map attribute -> (module, attribute)
_LAZY: Dict[str, Tuple[str, str]] = {
    "CoMomentConfig": (".estimators", "CoMomentConfig"),
    "cokurt_beta": (".estimators", "cokurt_beta"),
    "coskew_beta": (".estimators", "coskew_beta"),
    "panel_betas": (".estimators", "panel_betas"),
    "rolling_beta_series": (".estimators", "rolling_beta_series"),
    "crash_score": (".signal", "crash_score"),
    "load_universe": (".dataio", "load_universe"),
}

def __getattr__(name: str) -> Any:
    """Lazily resolve re-exported attributes."""
    try:
        module_name, attr_name = _LAZY[name]
    except KeyError as e:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from e

    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache for next access
    return value

def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
