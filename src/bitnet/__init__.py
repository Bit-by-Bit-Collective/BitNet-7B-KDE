# src/bitnet/__init__.py
"""
BitNet-7B-KDE core package.

Public API:
- Models: MiniBitNet, BitNetBlock, BitNetAttention, BitNetMLP, BitLinear,
          FakeQuantActivation, RMSNorm
- Losses: kd_cross_entropy_loss, format_loss, combined_loss
- Data:   KDTraceDataset, build_collate_fn (if available)

This package layout assumes your code lives under `src/bitnet/` and your
scripts import via `from bitnet import MiniBitNet, KDTraceDataset, ...`.
"""

from __future__ import annotations

# Package version (best-effort; falls back when not installed as a package)
try:  # Python ≥3.8
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except Exception:  # pragma: no cover
    version = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore

try:  # pragma: no cover
    __version__ = version("bitnet") if version else "0.0.0"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# --- Re-exports: models ---
from .models import (  # noqa: F401
    MiniBitNet,
    BitNetBlock,
    BitNetAttention,
    BitNetMLP,
    BitLinear,
    FakeQuantActivation,
    RMSNorm,
)

# --- Re-exports: losses ---
from .losses import (  # noqa: F401
    kd_cross_entropy_loss,
    format_loss,
    combined_loss,
)

# --- Re-exports: data (build_collate_fn may be optional) ---
try:
    from .data import KDTraceDataset, build_collate_fn  # type: ignore  # noqa: F401
    _HAS_BUILD_COLLATE = True
except Exception:  # pragma: no cover
    from .data import KDTraceDataset  # type: ignore  # noqa: F401
    build_collate_fn = None  # type: ignore
    _HAS_BUILD_COLLATE = False

__all__ = [
    "__version__",
    # models
    "MiniBitNet",
    "BitNetBlock",
    "BitNetAttention",
    "BitNetMLP",
    "BitLinear",
    "FakeQuantActivation",
    "RMSNorm",
    # losses
    "kd_cross_entropy_loss",
    "format_loss",
    "combined_loss",
    # data
    "KDTraceDataset",
    "build_collate_fn",
]
