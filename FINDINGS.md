# Multiverse data reconnaissance — findings

Real-data reconnaissance for the per-particle interpolation project (KASI internship,
Dr. S. E. Hong). Everything below was verified against the actual Multiverse data on
the KIAS `grammar` cluster (2026-07).

## 1. Environment / access
- **SSH**: `ssh hyunoh_yeo@grammar.kias.re.kr` (IP 210.107.146.172). VPN required first
  (AhnLab TrusGuard SSL VPN, server KIAS-CAC 210.107.146.50:443).
- OS: Rocky Linux 9.6, **x86_64 → little-endian**.
- Clusters active: grammar/lexicon/syntax (GPU), baekdu (MPI/CPU), etc.
- **Both datasets accessible**: DM snapshots + MBP merger tree.
- ⚠️ grammar default `python3` has **only numpy** (no scipy/matplotlib/cupy/torch/
  gpytorch) → built a venv `~/cosmo-env` + `~/setup_cosmo.sh` (numpy, scipy,
  matplotlib, camb, cosmoprimo, pypower). `cupy` not yet installed.
- ⚠️ `setup_cosmo.sh` must `unset I_MPI_PMI_LIBRARY` — the module system points it at
  `/usr/lib64/libpmi2.so`, which aborts pypower's MPI init on the login node
  (`PMI2_Job_GetId returned 14`).
- ⚠️ The PyPI package named `pypower` is an **unrelated electrical power-flow solver**.
  The cosmology one is `git+https://github.com/cosmodesi/pypower`.

## 2. Data location / structure
- **DM snapshots**: `/gpfs/mhee7173_snu/Multiverse_CPL` and
  `/multiverse/mhee7173_snu/MultiverseCPL` (**data being migrated gpfs → multiverse**).
  - per cosmology: `MV_Om<Om>_<w0>_<wa>/`
  - snapshot files: `<prefix>.<step:05d><subfile:05d>` (e.g. `INITIAL.0118000000`)
  - **250 subfiles per snapshot** (Nid=250), each ~1.1 GB, ~34 M particles
- **Use `SyncINITIAL` as the interpolation grid** (revised — see below). A cosmology
  directory holds hundreds of files (per-step `log*`, `params.*`, executables,
  `xzslice.*`, ...); the actual particle snapshots are only the `<prefix>.SSSSS?????`
  ones, in 250-subfile sets. Two prefixes carry particles:
  - **`SyncINITIAL`** — a **standardised ladder of the same 11 steps in every
    cosmology**: `00001 00281 00345 00441 00601 00921 01090 01241 01561 01706 01881`.
    Verified identical across `MV_Om0.26_-1.2_-0.8` and `MV_Om0.26_-1.0_-1.6`.
    `01881 = z=0`, `00001 = initial conditions` (both from headers, §5). This is the
    common ladder cross-cosmology work needs.
  - **`INITIAL`** — ad-hoc dumps at cosmology-dependent steps (e.g. 1150+1660 for one
    cosmology, 1150+1650 for another). Our earlier `INITIAL.01150` runs happened to
    coincide; do not rely on `INITIAL` steps matching across cosmologies.
  - (`PreFoF` seen earlier is FoF prep, not used here.)
  - ⚠️ Earlier guess "`SyncINITIAL` is index-sorted" is **wrong** — its `indx` is not
    ascending, and its subfiles are Eulerian slabs (§3), so cross-cosmology subfile
    membership differs.
- **MBP**: `/gpfs/mhee7173_snu/MultiverseMBPGalaxy/Om<X>/w0wa_<w0>_<wa>/` →
  `MergingTree.dat` + **`nstep_redshift.dat`** (authoritative step→z table).
- Cosmology-dir naming is slightly inconsistent (`-1` vs `-1.0`, `0` vs `0.0`);
  MBP uses `Om<X>/w0wa_..` while DM uses `MV_Om<X>_..`.

## 3. Snapshot binary format (verified on real data)

### File layout
A snapshot subfile is:
```
[ ASCII header  ~2923 B (text) ][ particle records  32 B each, back-to-back ]
```
- Header is plain text, one `define KEY = VALUE` per line, terminated by the line
  `#End of Ascii Header`. Read line-by-line until that marker; the byte offset just
  after it is where the binary begins (2923 B for the sample; varies).
- After the header: `Np` particle records with **no separators / no newlines**.
  N-th particle is at byte offset `header_bytes + N*32`.
- `Np` (this subfile) ≈ 34.35 M; file size ≈ 1.1 GB (header is 0.0003% of the file).

### One record = 32 bytes, LITTLE-ENDIAN
| field | type | bytes | note |
|-------|------|-------|------|
| x, y, z    | float32 ×3 | 12 | **displacement from the Lagrangian grid node** (grid units), not absolute |
| vx, vy, vz | float32 ×3 | 12 | raw code-unit velocity (tiny, ~1e-3) |
| indx       | int64      | 8  | GLOBAL Lagrangian index 0 .. Nx*Ny*Nz−1 (the particle's **identity**) |

**`indx` is a Lagrangian identity, x/y/z are a displacement from it — confirmed.**
In `SyncINITIAL.01881` (z=0) the Lagrangian iz decoded from `indx` spans the full
0..2047, yet the decoded z-*positions* of those same particles fall in a thin
0.00..4.30 cMpc/h band. That is only consistent if the stored float is an offset
from the node (wrapped periodically), and `indx` is the fixed identity. This is the
premise the whole project rests on, and it is also independently corroborated by the
real-data P(k) coming out physically sensible (§7).

Worked example (first real particle, `MV_Om0.26_-1_0/INITIAL.0118000000`):
```
x= 3.1298  -> A5 4E 48 40      (IEEE-754 float32, LE)
y=-22.3596 -> 76 E0 B2 C1
z=-2.2850  -> 71 3D 12 C0
v (small)  -> 17 B7 D1 39 | 17 B7 51 B9 | 17 B7 D1 B8
indx=37752980 -> 94 10 40 02 00 00 00 00   (0x02401094, LE = low byte first)
full 32 B: A5 4E 48 40 76 E0 B2 C1 71 3D 12 C0 17 B7 D1 39 17 B7 51 B9 17 B7 D1 B8 94 10 40 02 00 00 00 00
```
**Little-endian, no byte swap needed** (grammar is x86; data is LE — verified: the
big-endian interpretation yields garbage, LE yields valid positions/indices).

### Reading it (NumPy)
```python
REC = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                ('vx','<f4'),('vy','<f4'),('vz','<f4'),('indx','<i8')])  # <f4=LE float32, <i8=LE int64
rec = np.fromfile(f, dtype=REC)     # after seeking past the header; slices the 1.1 GB into a table
```

### Geometry / decode
- `Nx=Ny=Nz=2048`, `Mx=My=Mz=2048`, `MxMy=4194304`, `boxsize=1024 cMpc/h`,
  ngrid = 2048³ = 8,589,934,592.
- **Grid cell from indx**: `ix = indx % mx`, `iy = (indx % mxmy) // mx`, `iz = indx // mxmy`.
- **Absolute position**: `pos = fmod(offset + gridcell, N) * (boxsize/N)` → [0, 1024]. Validated.
- **subfiles are EULERIAN z-slabs** (by present-day position, not by identity).
  `SyncINITIAL.01881` subfile 0 = z-position ∈ [0.00, 4.30] cMpc/h, but its `indx`
  (hence Lagrangian iz) spans the whole 0..2047, scattered. **Consequence**: which
  particles land in a given subfile *differs per cosmology* — subfile 0 of two
  cosmologies had Np = 34,288,921 vs 34,297,322, and only 99.60 % of one's IDs were
  in the other's subfile 0 (the rest drifted into an adjacent slab). So **matching
  particles within one subfile silently drops the biggest movers** — see §5b.
- **Cross-cosmology alignment is by `indx` identity** (never by subfile position).
  The unbiased full-box method (`verify_shared_ic.py`, "window" mode) exploits that a
  particle moves ≪ a slab thickness, so A's slab j maps into B's slabs j±w (periodic
  in z); searching that window matches every particle by ID with ~2w+1 slabs resident.
- **Slab layout VERIFIED** (`check_subfiles.py`, `SyncINITIAL.01881`, all 250):
  subfile number tracks z monotonically, coverage 0.00..1024.00 cMpc/h, slab thickness
  3.8–4.3, and consecutive slabs meet to 3 decimals — **max gap 0.000, max overlap
  0.000**. So "subfile j = z-layer j" is not an assumption but a measured fact, and the
  window method's j±w reach is well-founded (±3 → ±12.9 cMpc/h, vs a max mover of 2.9).
- ⚠️ **the window bounds |Δz|, not |Δr|.** Slabs are cut along z, so a particle moving far
  in x/y but little in z is still matched and its full 3-D |Δr| measured exactly (this is
  why a max above w×4.3 can legitimately be reported). Only |Δz| > w×4.3 goes unmatched.
  Motion is near-isotropic (|Δz| ~ |Δr|/√3), so big movers usually are caught — but any
  run with unmatched > 0 has an **incomplete tail**, and needs a larger `--window`.
- **Velocities**: raw code units (~1e-3) → km/s conversion factor (Pfact=51.2 / Fact1 /
  Fact2 + scale factor) still TO BE DETERMINED from GOTPM source.

## 4. Coverage (three datasets, both storage paths)

| dataset | /gpfs | /multiverse | union / 56 | notes |
|---------|-------|-------------|-----------|-------|
| **DM snapshot** | 16 | 36 | **52** | `MV_Om*/`, paths **mostly disjoint but NOT fully** (migration in progress) |
| **MBP (merger tree)** | 57 | — | **57** | `Om*/w0wa_*/`, **complete** (56 sims + 1 `s09` σ8=0.9 variant); /gpfs only |
| **FoF (halo)** | 3 | 38 | **39** | `Om*/w0wa_*/`, ~17 missing — Minseong deleted bulky FoF intermediates, kept final merger trees (re-generable) |

- DM counting only `INITIAL.*` gave a misleading **21** — many cosmologies are stored
  as `SyncINITIAL` / `PreFoF` instead; counting all prefixes gives 52.
- **Search BOTH paths, and count particle files — a directory can be an empty shell.**
  Earlier note "paths are disjoint, each cosmology in one path" is **not quite right**:
  some names exist in both (`MV_Om0.26_-1.0_-1.6`, `_-0.6_0`, `_-1.4_0`). Worse, the
  `/gpfs` copy of `MV_Om0.26_-1.0_-1.6` has **only logs + executables, no particle
  data** — the real snapshot is the `/multiverse` copy. So `ls -d <dir>` succeeding
  proves nothing; check `ls <dir>/<prefix>.SSSSS????? | wc -l == 250`.
- ⚠️ **`MV_Om0.26_-1_0_s09`** (also the MBP `s09`) is the σ8-normalisation variant —
  **exclude from the interpolation grid** (§5b / param_space figure).
- Completeness: MBP (57, complete) > DM (52) > FoF (39). For phase-1 (DM interpolation)
  only the DM snapshots matter → 52/56 is enough. FoF/MBP are phase-2 (galaxies).
- FoF/MBP files use the merger-tree snapshot numbering (139, 162, 184, 205, …), the
  same as `nstep_redshift.dat`.

## 5. Redshift mapping (authoritative)
- `nstep_redshift.dat` = step→z table (133 snapshots; **step 1881 = z=0**, Nstep=1881).
  Confirms the `z = Amax/Anow − 1` estimate (Amax=48).
- Our snapshot steps → z:
  - **low-z group**: 1140→0.63, 1150→0.615, 1180→0.575, 1200→0.55, 1240→0.50,
    1270→0.47  ⇒ **spread z ≈ 0.47–0.63 (NOT a clean 0.5)**
  - **high-z group**: 1550→0.21, 1660→0.13, 1670→0.12, 1855→0.011
- Different cosmologies dump at different steps ⇒ their "low-z snapshot" is at
  genuinely different redshifts.
- **OPEN**: is step→z universal or per-cosmology? (nstep_redshift.dat is stored
  per-cosmology — needs a quick cross-cosmology compare.) But note `SyncINITIAL`
  uses the **same 11 steps for every cosmology** (§2), and `01881` reads z=0 from the
  header in each — so on the Sync ladder a common z is available by construction, and
  growth-factor normalisation is only needed for the ad-hoc `INITIAL` dumps.

## 5b. Validations

### Four full-box scans — what actually drives particle displacement (2026-07-27)

All four are complete-box scans at z=0 (`SyncINITIAL.01881`, window ±3): 8.59 G
particles each, `unmatched` at most 35 (4e-9 of the box), so every tail is complete.

| scan | what differs | median | 99.99% | max | >10 cMpc/h | ΔD/D |
|------|--------------|--------|--------|-----|------------|------|
| `ic_z0` | w0,wa: (−1.2,−0.8) vs (−1.0,−1.6), Ωm 0.26 | **0.030** | 0.417 | 5.52 | **0** | **−0.28 %** |
| `om21_w0scan` | w0 only: −1.0 vs −1.4, Ωm 0.21 | **1.047** | 5.25 | 9.96 | **0** | 10.39 % |
| `om21v36_lcdm` | Ωm: 0.21 vs 0.36, w0=−1 | **2.512** | 9.55 | 19.87 | 443,621 | 14.05 % |
| `om21v36_w14` | Ωm: 0.21 vs 0.36, w0=−1.4 | **2.754** | 9.55 | 18.85 | 206,962 | 9.87 % |

ΔD/D is the difference in linear growth from z=47 to z=0 between the two cosmologies
(cosmoprimo/CAMB). Two things fall out, and they change how the project should proceed.

**1. The (w0, wa) axis is NOT intrinsically safe — `ic_z0`'s pair is nearly degenerate.**
Its two cosmologies differ by Δw0=+0.2, Δwa=−0.8, which in CPL (`w(a)=w0+wa(1−a)`) cross
at a≈0.75, so their expansion histories almost coincide: **ΔD/D = 0.28 %**. That, not the
choice of parameter, is why its median is 0.030. Change w0 alone by 0.4 and the median
jumps 35× to 1.047. The earlier "w0/wa is easy" reading was an artefact of that one pair.

**2. Dark-energy displacement obeys a clean scaling; Ωm does not.**
Normalising by the growth difference:

| | \|Δr\|median / (ΔD/D in %) |
|--|--------------------------|
| `ic_z0` (w0,wa) | **0.108** |
| `om21_w0scan` (w0) | **0.101** |
| `om21v36_lcdm` (Ωm) | 0.179 |
| `om21v36_w14` (Ωm) | 0.279 |

The two dark-energy runs agree to 7 % **across a 35× range in displacement** — i.e. along
w0/wa, displacement is simply proportional to the growth difference. The Ωm runs are
1.8–2.8× above that line, and are *inconsistent with each other*: their growth differences
are 14.1 % vs 9.9 % yet their medians are nearly equal (2.51 vs 2.75), so along Ωm the
displacement tracks **ΔΩm itself, not ΔD**.

**Interpretation**: dark energy only rescales the growth amplitude, so a growth-factor
normalisation absorbs it entirely. Ωm additionally changes the **shape** of the initial
power spectrum (the turnover k_eq ∝ Ωm·h) and the IC amplitude (§5b, σ8 spread 8.5 %
across Ωm), so the field is not merely rescaled — it is reshaped, and particles rearrange
by more than growth alone predicts.

⇒ **Growth normalisation should fully solve the w0/wa direction and only partly the Ωm
direction.** That is a concrete, testable prediction and the next thing to measure.

### Shared-IC assumption — CONFIRMED at z=0, complete box, zero loss (2026-07-27)

**The full scan finished: 8,589,934,592 particles compared = exactly 2048³, with
`0 unmatched`.** Every particle in the box was matched by Lagrangian ID; nothing was
dropped at a slab boundary and nothing fell outside the search window. This is the
definitive version of the measurement below.

Pair: Ωm = 0.26, (−1.2, −0.8) vs (−1.0, −1.6), z = 0 (`SyncINITIAL.01881`), window ±3.

| statistic | value |
|-----------|-------|
| median \|Δr\| | **0.0302** (1/16.5 of the 0.5 interparticle spacing) |
| 99.9% / 99.99% | 0.182 / 0.417 |
| **max** | **5.52** |
| > 1 cMpc/h | 43,361 (0.0005%) |
| > 5 cMpc/h | **4 particles** (5e-8 %) |
| **> 10 cMpc/h** | **0** |

⇒ Along (w0, wa) the premise holds with a **1.8× margin even on the single worst
particle**, and by a factor ~330 on the median.

**No periodic-boundary problem.** The 8-subfile smoke made the top movers look piled
up near coordinate 0; that was purely its truncated z-range (0–34 of 1024). In the full
scan the largest movers are spread through the box — (685, 1013, 264), (910, 418, 71),
(294, 821, 963), (529, 234, 946), (78, 290, 60), (559, 220, 833) — with no edge
preference.

**Sensitive locations are clustered on halos**: 6 of the top 20 sit within ~1 cMpc/h of
(685.3, 1013.3, 263.9); two more pairs at (294, 821, 963) and (910, 418, 71). So the
premise holds globally while a handful of collapsed structures are mildly
cosmology-sensitive — exactly the "대부분 비슷한데 특정 위치만 민감" picture.

### (superseded) first unbiased partial result
Do the same particle IDs sit at (nearly) the same place across cosmologies? Yes, by a
wide margin. The earlier "CONFIRMED" was on a biased measurement (matching *within one
subfile*, which drops the boundary-crossing big movers, §3) — so its max was a lower
bound. The unbiased **window scan** (`verify_shared_ic.py`) fixes that by matching on
`indx` identity across neighbouring slabs, dropping nothing.

Pair: Ωm=0.26, **(−1.2,−0.8) vs (−1.0,−1.6)** at **z=0** (`SyncINITIAL.01881`). Mean
interparticle spacing 0.5 cMpc/h.

| measurement | median | 99.99% | max | >5 | >10 | N |
|-------------|--------|--------|-----|----|-----|---|
| biased subfile-0 intersect | 0.031 | 0.369 | 2.08 | 0 | 0 | 34 M |
| **unbiased, 8-slab smoke** | **0.032** | **0.398** | **2.92** | **0** | **0** | 275 M |
| unbiased, full 250 slabs | — | — | — | — | — | (running) |

- Typical shift ~0.03 = **1/16 of the interparticle spacing**; 99.99% within 0.40.
- **Nothing exceeds 5 cMpc/h**, let alone Hong's 10 — a ≥3.4× margin on the max.
- The 8-slab smoke test reported 124 k "unmatched" (0.045%): an **artefact of
  truncation**, not real movers — all of it is at the two z-ends of the 8-slab chunk
  (the middle slabs added zero); the full periodic 250-slab run has no such ends.
- **Sensitive locations are real and clustered** (Hong's prediction): the top movers
  are not scattered — six of the top 20 sit at ≈(53, 649, 33) cMpc/h and four more at
  ≈(484, 817, 25). These are collapsed halos where a tiny IC/cosmology difference gets
  amplified to ~2.5 cMpc/h, while the rest of the box barely moves. So the premise
  holds globally, with a handful of dense spots being mildly cosmology-sensitive —
  exactly "대부분 비슷한데 특정 위치만 민감."

**Still to do**: (1) the full 250-slab run (firms up the true max, kills the
truncation artefact); (2) the harder **different-Ωm** case (e.g. 0.21 vs 0.36), where
growth histories diverge most — this pair shares Ωm, so its small effect is expected.

### Growth factor — in-house ODE validated, then replaced by cosmoprimo (2026-07)
An in-house scipy ODE integration of the CPL growth equation was cross-checked
against cosmoprimo (CAMB engine) over 6 cosmologies × z=0…2: worst relative error
**D(z)/D(0) 0.029 %**, **f(z) 0.069 %** (0.000–0.005 % at our working z≈0.12–0.615).
The residual matched the expected size of neglecting radiation — not a bug.
Since the results were equivalent, the hand-rolled physics was **deleted** and
`cpl_growth.py` now wraps cosmoprimo directly (one code path, no dead code).

**Engine gotchas (important):**
- `engine='camb'` — correct CPL (w0, wa) physics, but `growth_factor` returns
  **NaN above z ≈ 200** (tabulation limit; `z_pk`/`extra_params` do NOT extend it).
- `engine='eisenstein_hu'` (and `bbks`) — works at any z but **silently ignores wa**
  (D(z) for wa=0 and wa=−1.6 differ by <0.05 %, vs 7.6 % with CAMB). **Do not use**
  for this project — wa is a primary Multiverse parameter.
- `engine='class'` would handle both but needs `pyclass` compiled (not installed).
⇒ We use **camb**, and set the mock's reference epoch to the simulations' actual
`z_init = 47` (was 1090, which is outside CAMB's range anyway). Real-pipeline
redshifts (z ≈ 0.1–0.6) are comfortably inside the valid range.

## 6. Implications for interpolation
- Reader works → can read real snapshots.
- **Grid = `SyncINITIAL` ladder** (11 common steps, same in every cosmology); align
  particles across cosmologies **by `indx` identity**, never by subfile position.
- **Redshift**: on the Sync ladder a common step (e.g. 01881 = z=0) already gives a
  common z. For the ad-hoc `INITIAL` dumps (different z per cosmology) still normalise
  via the linear growth factor (`mvinterp/cpl_growth.py`) before interpolating.
- Grid may be uneven in Om → possible request to Minseong for more extraction.

## 7. Code built

### Reading / inspection
- `gotpm.py` — validated GOTPM snapshot reader (header + little-endian records +
  indx→position decode; per-subfile / streaming / sort-by-index). **The single home
  of the format**: `read_records` (stride/id_mod/subsample) + `decode_positions`.
  Every analysis script imports from here — no duplicated readers.
- `mvinterp/inspect_snapshot.py` — header + endianness diagnostic (deliberately
  independent of `gotpm.py`: it brute-forces both byte orders to *validate* the format,
  so it must not import the reader it validates).
- `scan_snapshots.py` — coverage / redshift scan.
- `compare_particles.py` — cross-cosmology |Δr| on ONE subfile (the biased first cut).
- `verify_shared_ic.py` — **unbiased full-box** |Δr| by the window method: streams all
  250 slabs, matches by `indx` across ±w neighbours, reports percentiles + threshold
  counts + the top-20 movers' coordinates. Supersedes `compare_particles.py`.
- `check_subfiles.py` — certifies the slab layout (ordered / contiguous / no overlap)
  and dumps each subfile's z-histogram to `.npz`; `plot_subfiles.py` renders it (per-
  subfile z curves + a [subfile × z] image whose diagonal band shows clean ordering).

### Analysis (all on standard libraries — see below)
- `snapshot_view.py` — z-slab → 2D density map via **pypower `CatalogMesh`**
  (`--mas ngp|cic|tsc`), ASCII cosmic web + `.npy` + `--png`.
- `power_spectrum.py` — P(k) via **pypower `CatalogFFTPower`**. Needs the whole
  box (all 250 subfiles); `--stride N` mmap-subsamples to cut I/O ~N/128×.
- `plot_pk.py` — overlay P(k) of several cosmologies + their ratio (the ratio is the
  quantity an interpolated snapshot must reproduce).
- `mvinterp/cpl_growth.py` — D(z), f(z) via **cosmoprimo** (CAMB engine).

### First real-data P(k) (2026-07)
`SyncINITIAL`-pair work aside, ran `power_spectrum.py` on `INITIAL.01150` (z=0.615) of
`MV_Om0.26_-1.2_-0.8`, `--stride 1280` (6.71 M particles), nmesh 512, TSC+interlacing.
Sanity checks all pass: **shot noise 160.0 = box³/N exactly**; the empty-box artefact
at low k (8×10⁷ from a 2-subfile smoke test) collapsed to a physical 1.9×10⁴; P(k)
falls monotonically 1.9×10⁴ → 198 over k=0.009→1.56. Trustworthy for **k ≲ 0.3**
(P ≫ shot noise); higher k needs a smaller stride. First physical quantity measured
from the real data — and an independent check that the `indx`→position decode is right.

### Interpolation
- `mvinterp/gpu_interp.py` — CuPy, tiled over the particle axis (the only
  hand-written numerics; SciPy is CPU-only and cannot tile 2048³).
- `mvinterp/gp_interp.py` — GPyTorch GP (learned hyperparameters + uncertainty).
- `mvinterp/compare.py` — CPU 4-way comparison on **scipy.interpolate**
  (`LinearNDInterpolator`, `RBFInterpolator`), RBF shape parameter by
  leave-one-cosmology-out CV.

### Library policy
Everything that a community package does well is delegated to it — pypower
(mesh + P(k)), cosmoprimo (cosmology), scipy (scattered interpolation), GPyTorch
(GP). **nbodykit is deliberately not used**: last release 0.3.16 (~2021), effectively
unmaintained, and superseded by the cosmodesi stack we already depend on.
Hand-written code is confined to the GOTPM reader (no library reads this format)
and the GPU/tiled interpolator (no library does per-particle tiling at 2048³).

### Mock validation of the interpolators
`python -m mvinterp.compare --n 48` → 48³ particles × 36 mock cosmologies,
median per-particle position error [cMpc/h] (rms displacement 8.1):

| test | linear | quadratic | RBF | GP |
|------|--------|-----------|-----|-----|
| off-grid (0.285, −0.85, 0.30) | 0.590 | 0.228 (2.6×) | **0.072 (8.2×)** | 0.270 (2.2×) |
| leave-one-out (0.26, −1.0, 0.0) | 0.031 | 0.080 (0.4×) | **0.0008 (42×)** | 0.014 (2.2×) |

⇒ RBF beats linear by ~8× off-grid and ~40× at a held-out node; quadratic helps
off-grid but *hurts* at a node (a global fit is not node-exact, unlike linear/RBF).
GP is competitive and additionally returns a posterior std (0.02–0.04), which is
what makes it useful for flagging where the grid is too sparse.
**These are mock numbers** — the real-data equivalent is still to be run.

## 8. Still pending
- **Full 250-slab |Δr| at z=0** (`verify_shared_ic.py`) — running; the 8-slab smoke
  already gives max 2.92 / >10 = 0, this firms up the true max and removes the
  truncation artefact.
- **Different-Ωm pair** |Δr| (e.g. Om0.21 vs Om0.36 on `SyncINITIAL.01881`) — the
  hardest case, currently untested. First confirm both have the Sync 01881 step.
- **Which `SyncINITIAL` steps exist across all 52 cosmologies** (the usable grid).
- **step→z for the 11 Sync steps** (from `nstep_redshift.dat`; only 01881=z=0 nailed).
- **Velocity unit** conversion factor (from GOTPM source).
- **P(k) ratio P_A/P_B on real data** as the interpolation accuracy metric (single
  spectra now work; need a second cosmology at a matching step — use `SyncINITIAL`).
- Run the interpolators on **real snapshots** (aligned by indx; common Sync step).
- `cupy` install on grammar for the GPU path.

## Quick commands
```bash
# read a real subfile with the reader
python3 gotpm.py /gpfs/mhee7173_snu/Multiverse_CPL/MV_Om0.26_-1_0/INITIAL.0118000000
# header + endianness
python3 mvinterp/inspect_snapshot.py <subfile> --raw
# is a directory a real snapshot or an empty shell? (expect 250)
ls <cosmo_dir>/SyncINITIAL.01881????? | wc -l
# unbiased full-box shared-IC test at z=0
python3 verify_shared_ic.py <DIR_A> <DIR_B> --prefix SyncINITIAL --step 1881
# real-data P(k) (whole box, 1/10 I/O)
python3 power_spectrum.py <cosmo_dir> --step 1881 --prefix SyncINITIAL --stride 1280 --nmesh 512
# authoritative redshift of a step
awk '$1==1200{print $2}' /gpfs/mhee7173_snu/MultiverseMBPGalaxy/Om0.31/w0wa_-1_0/nstep_redshift.dat
```
