"""Linear growth D(z), growth rate f(z) and E(z) for flat CPL (w0-wa) cosmology.

Backed by **cosmoprimo** (CAMB engine) — the validated community standard.

An earlier in-house scipy ODE integration was cross-checked against this and agreed
to 0.029 % in D(z)/D(0) and 0.069 % in f(z) over Om 0.21-0.36, z 0-2 (see
FINDINGS.md). Since the results are equivalent, the hand-rolled physics was removed
in favour of the library.
"""
import numpy as np
from cosmoprimo import Cosmology


class CPLGrowth:
    """Growth quantities for a flat w0-wa CDM cosmology. D is normalised to D(0)=1."""

    def __init__(self, Om0, w0=-1.0, wa=0.0, h=0.72):
        self.Om0, self.w0, self.wa, self.h = float(Om0), float(w0), float(wa), float(h)
        self._cosmo = Cosmology(Omega_m=self.Om0, h=self.h,
                                w0_fld=self.w0, wa_fld=self.wa, engine="camb")
        self._D0 = float(self._cosmo.growth_factor(0.0))

    def D(self, z):
        """Linear growth factor, normalised so D(0) = 1."""
        return np.asarray(self._cosmo.growth_factor(z), dtype=float) / self._D0

    def f(self, z):
        """Growth rate f = dlnD/dlna."""
        return np.asarray(self._cosmo.growth_rate(z), dtype=float)

    def H_over_H0(self, z):
        """E(z) = H(z)/H0."""
        return np.asarray(self._cosmo.efunc(z), dtype=float)
