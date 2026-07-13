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
  gpytorch) → need a conda env / `module load` before interpolation + plotting.

## 2. Data location / structure
- **DM snapshots**: `/gpfs/mhee7173_snu/Multiverse_CPL` and
  `/multiverse/mhee7173_snu/MultiverseCPL` (**data being migrated gpfs → multiverse**).
  - per cosmology: `MV_Om<Om>_<w0>_<wa>/`
  - snapshot files: `<prefix>.<step:05d><subfile:05d>` (e.g. `INITIAL.0118000000`)
  - **250 subfiles per snapshot** (Nid=250), each ~1.1 GB, ~34 M particles
  - **3 prefixes**: `INITIAL` (raw), `SyncINITIAL` (what Hong's example
    `makeUnsmoothDensGrid` read — likely index-sorted), `PreFoF` (FoF prep)
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
| x, y, z    | float32 ×3 | 12 | in-cell **offset** (grid units), not absolute |
| vx, vy, vz | float32 ×3 | 12 | raw code-unit velocity (tiny, ~1e-3) |
| indx       | int64      | 8  | GLOBAL Lagrangian index 0 .. Nx*Ny*Nz−1 |

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
- **subfiles are z-slab decomposed** (subfile 0 = z∈[0,~4.5], Local_nz=9). So one subfile
  is a thin z-slab; `indx` within it still spans the whole box (0..2048³−1), scattered.
- **Cross-cosmology alignment needs sort by `indx`** (subfile membership differs per
  cosmology) — unless `SyncINITIAL` is already index-sorted (to check).
- **Velocities**: raw code units (~1e-3) → km/s conversion factor (Pfact=51.2 / Fact1 /
  Fact2 + scale factor) still TO BE DETERMINED from GOTPM source.

## 4. Coverage
- **52 / 56 cosmologies have snapshots** (counting all 3 prefixes).
  (Counting only `INITIAL.*` gave a misleading 21 — many are stored as
  SyncINITIAL / PreFoF.)
- Migration in progress, so this is fluid. Per-Om distribution still to be re-checked
  with a full prefix×step scan.

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
  per-cosmology — needs a quick cross-cosmology compare.) If universal → snapshots
  are at different z → **growth-factor normalisation to a common z is required**.

## 6. Implications for interpolation
- Reader works → can read real snapshots.
- **Alignment**: sort by global indx (or use SyncINITIAL if pre-sorted).
- **Redshift spread** → normalise to a common z via the linear growth factor
  (`mvinterp/cpl_growth.py`) before interpolating in (Om,w0,wa). This is where the
  physics-informed step becomes essential, not optional.
- Grid may be uneven in Om → possible request to Minseong for more extraction.

## 7. Code built
- `mvinterp/gotpm.py` — validated GOTPM snapshot reader (header + little-endian
  records + indx→position decode; per-subfile / streaming / sort-by-index).
- `mvinterp/inspect_snapshot.py` — header + endianness diagnostic.
- `scan_snapshots.py` — coverage / redshift scan.
- Interpolation (local, mock-validated): linear / quadratic / RBF / GP in
  `mvinterp/gpu_interp.py` + `gp_interp.py`; `compare.py` runs the 4-way on CPU.

## 8. Still pending
- DM **prefix×step scan** (which prefix covers all 52; which step has the most
  cosmologies = the interpolation grid).
- **step→z universal vs per-cosmology** test.
- **Velocity unit** conversion factor (from GOTPM source).
- **Snapshot visualisation** (slice density map, similar cosmologies → similar
  structure) — meeting deliverable.
- grammar **env setup** (cupy / torch / matplotlib).

## Quick commands
```bash
# read a real subfile with the reader
python3 mvinterp/gotpm.py /gpfs/mhee7173_snu/Multiverse_CPL/MV_Om0.26_-1_0/INITIAL.0118000000
# header + endianness
python3 mvinterp/inspect_snapshot.py <subfile> --raw
# authoritative redshift of a step
awk '$1==1200{print $2}' /gpfs/mhee7173_snu/MultiverseMBPGalaxy/Om0.31/w0wa_-1_0/nstep_redshift.dat
```
