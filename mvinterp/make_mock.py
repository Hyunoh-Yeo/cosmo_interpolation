"""Generate physically-motivated mock 'Multiverse' snapshots.

Every Multiverse run shares the same initial conditions (same Fourier phases,
matched sigma8 at z~1090), so particle ID i is the *same* Lagrangian particle in
every cosmology. We mimic that here:

    x(theta) = q + s * n1(theta) * Psi1  -  s * c2 * n1(theta)^2 * Psi2     (2LPT)
    v(theta) = a H(a;theta) [ f1 * disp1  +  f2 * disp2 ]                   (km/s)

where (q, Psi1, Psi2) are FIXED across cosmologies (the shared IC) and the only
theta = (Om0, w0, wa) dependence enters through the linear growth amplitude
n1(theta) = [D(z)/D(z_norm)] (relative to the grid mean) and growth rate f(theta).
Because D and f are nonlinear in theta, so are x and v -- which is exactly what
makes per-particle interpolation across cosmologies a non-trivial test.

This is a deliberate LPT-level simplification (no shell crossing / nonlinear
collapse) so the whole pipeline runs on a laptop. On `grammar` you swap this out
for the real GOTPM snapshot reader; the interpolation code is unchanged.
"""
import numpy as np
from .cpl_growth import CPLGrowth

Z_NORM = 1090.0  # common initial-condition epoch


def generate_fields(n=32, boxsize=1024.0, seed=42, ns=1.0, smooth_cells=1.0):
    """Build the shared IC: Lagrangian grid q and 1LPT/2LPT displacement fields.

    Returns q, Psi1, Psi2 each of shape (n^3, 3), float32. Psi1/Psi2 are
    normalised to unit rms vector magnitude (absolute scale set later by `s`).
    """
    rng = np.random.default_rng(seed)

    k1d = np.fft.fftfreq(n, d=1.0) * n          # integer wavenumbers
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    kfac = 2.0 * np.pi / boxsize
    Kx, Ky, Kz = kx * kfac, ky * kfac, kz * kfac
    K2 = Kx**2 + Ky**2 + Kz**2
    K2[0, 0, 0] = 1.0
    inv = 1.0 / K2
    inv[0, 0, 0] = 0.0

    # Gaussian random linear density field with P(k) ~ k^ns and a grid-scale cutoff.
    delta_k = np.fft.fftn(rng.standard_normal((n, n, n)))
    kmag = np.sqrt(K2)
    R = smooth_cells * boxsize / n
    amp = np.where(kmag > 0, kmag ** (0.5 * ns) * np.exp(-0.5 * (kmag * R) ** 2), 0.0)
    delta_k *= amp
    delta_k[0, 0, 0] = 0.0

    def disp_from_source(src_k):
        out = np.empty((n, n, n, 3))
        for j, Kj in enumerate((Kx, Ky, Kz)):
            out[..., j] = np.fft.ifftn(1j * Kj * src_k * inv).real
        return out

    # 1LPT (Zel'dovich) displacement: Psi1_j = i k_j delta / k^2
    Psi1 = disp_from_source(delta_k)

    # 2LPT source: delta2 = sum_{i<j} (phi,ii phi,jj - phi,ij^2), phi,ij = k_i k_j delta / k^2
    def d2(Ki, Kj):
        return np.fft.ifftn(Ki * Kj * delta_k * inv).real
    pxx, pyy, pzz = d2(Kx, Kx), d2(Ky, Ky), d2(Kz, Kz)
    pxy, pxz, pyz = d2(Kx, Ky), d2(Kx, Kz), d2(Ky, Kz)
    delta2 = pxx * pyy + pxx * pzz + pyy * pzz - pxy**2 - pxz**2 - pyz**2
    delta2_k = np.fft.fftn(delta2)
    delta2_k[0, 0, 0] = 0.0
    Psi2 = disp_from_source(delta2_k)

    # Lagrangian grid (cell centres), cMpc/h
    cell = boxsize / n
    g = (np.arange(n) + 0.5) * cell
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([qx, qy, qz], axis=-1).reshape(-1, 3)

    Psi1 = Psi1.reshape(-1, 3)
    Psi2 = Psi2.reshape(-1, 3)
    Psi1 /= np.sqrt((Psi1**2).sum(1).mean())
    Psi2 /= np.sqrt((Psi2**2).sum(1).mean())
    return q.astype(np.float32), Psi1.astype(np.float32), Psi2.astype(np.float32)


def _amp_table(thetas, z):
    """Per-cosmology growth amplitude r=D(z)/D(z_norm), growth rate f, and E(z)."""
    r = np.empty(len(thetas))
    f = np.empty(len(thetas))
    E = np.empty(len(thetas))
    for i, (Om0, w0, wa) in enumerate(thetas):
        cg = CPLGrowth(Om0, w0, wa)
        r[i] = cg.D(z) / cg.D(Z_NORM)
        f[i] = cg.f(z)
        E[i] = cg.H_over_H0(z)
    return r, f, E


def catalog_from_amp(q, Psi1, Psi2, r, f, E, z, r_ref,
                     sigma_disp=8.0, c2=0.15, boxsize=1024.0):
    """Build one (pos, vel) catalog from precomputed growth scalars."""
    n1 = r / r_ref
    disp1 = sigma_disp * n1 * Psi1
    disp2 = -sigma_disp * c2 * (n1**2) * Psi2
    pos = q + disp1 + disp2
    pos = np.mod(pos, boxsize)                      # periodic wrap

    aH = (1.0 / (1.0 + z)) * 100.0 * E              # km/s per (cMpc/h)
    f2 = 2.0 * f                                    # 2nd-order growth rate ~ 2f
    vel = aH * (f * disp1 + f2 * disp2)
    return pos.astype(np.float32), vel.astype(np.float32)


def build_dataset(thetas, z=0.0, n=32, boxsize=1024.0, seed=42,
                  sigma_disp=8.0, c2=0.15):
    """Return a dict with the shared IC and per-cosmology pos/vel catalogs.

    keys: theta (M,3), pos (M,N,3), vel (M,N,3), q (N,3), Psi1, Psi2,
          z, boxsize, r_ref, and the growth tables r,f,E.
    """
    thetas = np.asarray(thetas, dtype=float)
    q, Psi1, Psi2 = generate_fields(n=n, boxsize=boxsize, seed=seed)
    r, f, E = _amp_table(thetas, z)
    r_ref = float(r.mean())

    M, N = len(thetas), q.shape[0]
    pos = np.empty((M, N, 3), dtype=np.float32)
    vel = np.empty((M, N, 3), dtype=np.float32)
    for i in range(M):
        pos[i], vel[i] = catalog_from_amp(q, Psi1, Psi2, r[i], f[i], E[i], z,
                                          r_ref, sigma_disp, c2, boxsize)
    return dict(theta=thetas, pos=pos, vel=vel, q=q, Psi1=Psi1, Psi2=Psi2,
                z=z, boxsize=boxsize, r_ref=r_ref, sigma_disp=sigma_disp, c2=c2,
                r=r, f=f, E=E)


def truth_at(ds, theta, z=None):
    """Generate the *true* catalog at an arbitrary (unsimulated) theta, using the
    same shared IC -- the held-out ground truth for interpolation tests."""
    z = ds["z"] if z is None else z
    r, f, E = _amp_table([theta], z)
    return catalog_from_amp(ds["q"], ds["Psi1"], ds["Psi2"], r[0], f[0], E[0],
                            z, ds["r_ref"], ds["sigma_disp"], ds["c2"], ds["boxsize"])
