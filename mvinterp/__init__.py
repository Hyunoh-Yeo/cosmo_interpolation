"""mvinterp: GPU-accelerated per-particle interpolation of Multiverse-style
N-body snapshots across (Om0, w0, wa) cosmologies.
"""
from .cpl_growth import CPLGrowth
from .make_mock import build_dataset, truth_at, generate_fields

# CuPy compute path (GPU nodes). Guarded so the package still imports on a laptop
# without CuPy -- the host-side pieces (mock, growth, GP, compare) then still work.
try:
    from .backend import xp, backend_name
    from .gpu_interp import CosmologyInterpolator, select_epsilon
except Exception:  # CuPy not installed
    xp = None
    def backend_name():
        return "cupy not available (CPU host only)"
    CosmologyInterpolator = None
    select_epsilon = None

# GP variant (needs torch + gpytorch).
try:
    from .gp_interp import GPCosmologyInterpolator
except Exception:
    GPCosmologyInterpolator = None

__all__ = [
    "xp", "backend_name",
    "CPLGrowth", "build_dataset", "truth_at", "generate_fields",
    "CosmologyInterpolator", "select_epsilon", "GPCosmologyInterpolator",
]
