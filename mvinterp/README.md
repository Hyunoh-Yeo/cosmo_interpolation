# mvinterp — GPU per-particle interpolation across cosmologies

Prototype for the KASI internship (Dr. S. E. Hong): reconstruct Multiverse N-body
snapshot data at **unsimulated** cosmologies `(Om0, w0, wa)` by interpolating each
particle across the cosmologies that *were* simulated.

The Multiverse set shares one set of initial conditions, so particle ID `i` is the
same Lagrangian particle in every run. We track how each particle's displacement
and peculiar velocity move with `(Om0, w0, wa)` and interpolate per-particle —
aiming to beat naive linear interpolation, as discussed.

**CUDA required.** The compute path is CuPy-only (no NumPy fallback) since it runs
on the `grammar` GPU cluster. cuBLAS/cuSOLVER *is* the optimal CUDA path for these
GEMMs/solves — hand-written `.cu` kernels would not be faster. Host-side data prep
(mock generation, growth ODE) uses NumPy/SciPy and is transferred to the device by
the interpolator.

## Methods

- **`linear` — the baseline (node-exact).** Proper linear interpolation on the
  scattered `(Om0,w0,wa)` nodes: Delaunay triangulation → barycentric blend of the
  ≤4 surrounding cosmologies. Reproduces the simulated cosmologies *exactly*,
  works on the real irregular Multiverse grid, falls back to the nearest node
  outside the convex hull. This is "simple linear interpolation" — the deliverable.
- **`rbf` — in development (the "better model").** Multiquadric RBF + linear
  polynomial tail; smooth, captures curvature the linear baseline misses.
  (Next: growth-factor-normalised interpolation, see project notes.)

## Why it's a good GPU workload — and how it scales to 2048³

Both methods end in one GEMM over the particle catalog:

```
Y(theta*) = qbasis(theta*) @ coeffs     # qbasis: (Q, k)
# linear: coeffs = Y         (k=M, barycentric weights in qbasis)
# rbf   : coeffs = L @ Y     (L: (M+4, M) reusable operator, built once)
```

The full coefficient matrix for the real `N = 2048³` catalog would be many TB and
never fits in GPU memory. The cosmology-only structure (triangulation / `L`) is
tiny, so we **never form it globally** — `predict()` streams the particle axis in
**tiles** (two cuBLAS GEMMs each). Each tile is independent → trivially shardable
across multiple GPUs.

## Files

| file | role |
|------|------|
| `backend.py`    | `xp = cupy` (CUDA required) + timing/memory helpers |
| `cpl_growth.py` | CPL `w0waCDM` linear growth `D(z)`, growth rate `f(z)` (host, SciPy ODE) |
| `make_mock.py`  | physically-motivated mock snapshots (shared IC + 2LPT; host) |
| `gpu_interp.py` | `CosmologyInterpolator`: per-particle RBF (+poly) and linear, tiled |
| `demo.py`       | build mock → interpolate → accuracy (RBF vs linear) + timing |

## Run (on the GPU node, `grammar`)

```bash
pip install cupy-cuda12x               # match the cluster CUDA toolkit (11x/12x)
python -m mvinterp.demo                # quick: 32^3 particles, 36 cosmologies
python -m mvinterp.demo --n 128 --plot # larger; saves error_diagnostic.png
python -m mvinterp.demo --n 256 --tile 4000000   # lower --tile if GPU OOM
```

### Representative accuracy (validated)

`linear` is the baseline deliverable; `rbf` is the in-development model
(median per-particle position error, mock data, n=48):

```
Test A  off-grid (0.285,-0.85,0.30):    linear 0.90  -> rbf 0.19  cMpc/h  (~4.7x)
Test B  leave-one-out interior point:   linear 0.030 -> rbf 0.0031 cMpc/h (~9.6x)
```

## API

```python
from mvinterp import CosmologyInterpolator
# theta_train: (M,3) = (Om0,w0,wa);  disp_train: (M,N,3) displacement = pos - q
ip = CosmologyInterpolator(method="rbf").fit_basis(theta_train)   # cheap, once
disp_star = ip.predict(disp_train, [[0.285, -0.85, 0.30]], tile=2_000_000)[0]
```

Interpolate **displacement** `pos - q` (and velocity), not absolute position, to
avoid the periodic-box wrap discontinuity; reconstruct with `pos = (q + disp) % L`.

## Using real Multiverse data instead of the mock

Swap the mock for the GOTPM snapshot reader (see `../samples/`). The interpolator
only needs, per cosmology, per-particle vectors aligned by particle ID. For the
full `2048³` run you typically won't hold all `M` catalogs in memory at once —
stream tiles from disk and call `predict_tile(Y_tile, theta_query)` directly:

```python
ip = CosmologyInterpolator(method="rbf").fit_basis(theta_train)
for Y_tile in read_particle_tiles(...):        # (M, t, 3) for a slab of particle IDs
    pred = ip.predict_tile(Y_tile, theta_query)  # (Q, t, 3) on GPU -> write out
```

## Known simplifications (mock only)

- Snapshots are 2LPT (no shell crossing / nonlinear collapse). Real snapshots are
  fully nonlinear, so the true `theta`-dependence is richer — but the per-particle,
  shared-IC interpolation strategy is identical.
- σ8 is matched at the IC epoch (`z_norm=1090`); present-day amplitude varies with
  cosmology through `D(z)/D(z_norm)`.
- The `(w0, wa)` grid here is a regular 4×3×3; the real set is irregular (4×13 +
  off-grid extras). RBF handles scattered nodes natively, so this carries over.
