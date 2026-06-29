"""mvinterp: GPU-accelerated per-particle interpolation of Multiverse-style
N-body snapshots across (Om0, w0, wa) cosmologies.

Prototype for the KASI internship (Dr. S. E. Hong): reconstruct snapshot data at
unsimulated cosmologies by exploiting the shared initial conditions of the
Multiverse set. Runs on CuPy (GPU, e.g. `grammar`) or NumPy (CPU) transparently.
"""
from .backend import xp, backend_name
from .cpl_growth import CPLGrowth
from .make_mock import build_dataset, truth_at, generate_fields
from .gpu_interp import CosmologyInterpolator

__all__ = [
    "xp", "backend_name",
    "CPLGrowth", "build_dataset", "truth_at", "generate_fields",
    "CosmologyInterpolator",
]
