"""Linear growth factor D(z) and growth rate f(z)=dlnD/dlna for a flat
w0-wa-CDM (CPL) cosmology.

CPL dark-energy equation of state:  w(a) = w0 + wa (1 - a)
Dark-energy density:  rho_DE(a)/rho_DE0 = a^{-3(1+w0+wa)} * exp(-3 wa (1-a))
Flat universe (no radiation):  E(a)^2 = Om0 a^-3 + (1-Om0) * rho_DE(a)/rho_DE0

The linear growth ODE in N = ln a:
    D'' + (2 + 0.5 dlnE2/dlna) D' - 1.5 Omega_m(a) D = 0
with Omega_m(a) = Om0 a^-3 / E(a)^2.

These run on the CPU (NumPy/SciPy) once per cosmology -- they are tiny and
feed the mock generator and (optionally) growth-normalised interpolation.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


class CPLGrowth:
    def __init__(self, Om0, w0=-1.0, wa=0.0, a_init=1e-5, n=4096):
        self.Om0 = float(Om0)
        self.w0 = float(w0)
        self.wa = float(wa)
        self.Ode0 = 1.0 - self.Om0

        lna = np.linspace(np.log(a_init), 0.0, n)
        # Matter-dominated initial conditions: D ∝ a  =>  D'=D in N=ln a.
        y0 = [np.exp(lna[0]), np.exp(lna[0])]
        sol = solve_ivp(self._rhs, (lna[0], lna[-1]), y0, t_eval=lna,
                        method="RK45", rtol=1e-9, atol=1e-12)
        D = sol.y[0]
        dDdN = sol.y[1]

        self._lna = lna
        self._D_spline = CubicSpline(lna, D)
        self._f_spline = CubicSpline(lna, dDdN / D)  # f = dlnD/dlna
        self._D0 = float(D[-1])                       # D at a=1 (unnormalised)

    # --- background ---
    def E2(self, a):
        a = np.asarray(a, dtype=float)
        de = a ** (-3.0 * (1.0 + self.w0 + self.wa)) * np.exp(-3.0 * self.wa * (1.0 - a))
        return self.Om0 * a ** -3 + self.Ode0 * de

    def _dlnE2_dlna(self, a):
        de = a ** (-3.0 * (1.0 + self.w0 + self.wa)) * np.exp(-3.0 * self.wa * (1.0 - a))
        dE2_da = (-3.0 * self.Om0 * a ** -4
                  + self.Ode0 * de * (-3.0 * (1.0 + self.w0 + self.wa) / a + 3.0 * self.wa))
        return a * dE2_da / self.E2(a)

    def _rhs(self, lna, y):
        a = np.exp(lna)
        D, dDdN = y
        Om_a = self.Om0 * a ** -3 / self.E2(a)
        ddDdN = -(2.0 + 0.5 * self._dlnE2_dlna(a)) * dDdN + 1.5 * Om_a * D
        return [dDdN, ddDdN]

    # --- growth, normalised so D(z=0)=1 ---
    def D(self, z):
        lna = np.log(1.0 / (1.0 + np.asarray(z, dtype=float)))
        return self._D_spline(lna) / self._D0

    def f(self, z):
        lna = np.log(1.0 / (1.0 + np.asarray(z, dtype=float)))
        return self._f_spline(lna)

    def H_over_H0(self, z):
        a = 1.0 / (1.0 + np.asarray(z, dtype=float))
        return np.sqrt(self.E2(a))

    def sigma8_ratio(self, z=0.0, z_norm=1090.0):
        """D(z)/D(z_norm): linear amplitude relative to the common IC epoch.

        All Multiverse runs share the same linear field at z~1090, so this ratio
        is exactly the cosmology-dependent factor multiplying the fixed IC field.
        """
        return self.D(z) / self.D(z_norm)


def growth_table(thetas, z, z_norm=1090.0):
    """Vectorised helper: return arrays (g1, f) for a list of (Om0,w0,wa) at redshift z.

    g1 = D(z)/D(z_norm) is the 1LPT amplitude relative to the IC epoch.
    """
    g1 = np.empty(len(thetas))
    f = np.empty(len(thetas))
    for i, (Om0, w0, wa) in enumerate(thetas):
        cg = CPLGrowth(Om0, w0, wa)
        g1[i] = cg.D(z) / cg.D(z_norm)
        f[i] = cg.f(z)
    return g1, f
