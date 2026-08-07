# Multiverse per-particle interpolation — status

**The verification phase is complete.** Six complete-box scans at z=0 (8.59 G particles
each) settle whether the shared-IC premise supports per-particle interpolation. It does —
and the measurements also say *why*, which shapes how the interpolation should be built.

---

## 1. The premise, and how it was tested

All Multiverse runs start from identical initial conditions, so a given `indx` is the same
Lagrangian particle in every cosmology. Interpolating across cosmologies only makes sense
if that particle stays in roughly the same place. The criterion from last meeting: **does
anything move more than 10 cMpc/h by z=0?**

**Method.** Match by `indx` between two cosmologies at z=0 and measure |Δr|. Subfiles are
Eulerian z-slabs, so a particle sits in *different* subfiles in the two runs; matching
within one subfile silently drops the boundary-crossers — exactly the biggest movers,
which is what the first attempt did. The scan instead searches A's slab j against B's
slabs j±w (periodic in z), so nothing is dropped. Every scan below covers
**8,589,934,592 particles = 2048³ = the entire box**, with ≤35 unmatched (4e-9).

---

## 2. Result: the Ωm axis is interpolatable
**[om_ladder.png]**

| ΔΩm | pair | median | max | >10 cMpc/h |
|-----|------|--------|-----|------------|
| **0.05** — the real grid spacing | 0.21→0.26 | **1.097** | 12.65 | **41** (4.8e-9) |
| 0.10 | 0.26→0.36 | 1.585 | 19.03 | 129,853 |
| 0.15 — corner to corner | 0.21→0.36 | 2.512 | 19.87 | 443,621 |

**(a) At the spacing interpolation actually uses, the criterion is met by ~8 orders of
magnitude** — 41 particles out of 8.59 billion. Every alarming number so far came from the
corner-to-corner case, which interpolation never has to handle.

**(b) The displacement field is coherent, not chaotic.** The 0.15 step is the composition
of the other two, which turns it into a test:

| | median \|Δr\| |
|--|-------------|
| 0.21→0.26 | 1.097 |
| 0.26→0.36 | 1.585 |
| quadrature sum — what *uncorrelated* steps would give | 1.927 |
| linear sum — what *fully coherent* steps would give | 2.681 |
| **measured 0.21→0.36** | **2.512** |

**94 % of the linear sum.** Particles keep being pushed the same way as Ωm increases. This
is the result that matters: large motion was never the risk — *unpredictable* motion would
have been, and the motion is not unpredictable.

**(c) Scaling is mildly sublinear with no saturation:** |Δr| ∝ ΔΩm^0.75.

---

## 3. Mechanism: displacement is set by the growth difference — for dark energy only
**[dr_scaling.png]**

| scan | what differs | median | max | >10 | ΔD/D | median per 1 % ΔD |
|------|--------------|--------|-----|-----|------|-------------------|
| (−1.2,−0.8) vs (−1.0,−1.6), Ωm 0.26 | dark energy | 0.030 | 5.52 | **0** | **−0.28 %** | **0.108** |
| w0: −1.0 vs −1.4, Ωm 0.21 | dark energy | 1.047 | 9.96 | **0** | 10.39 % | **0.101** |
| Ωm 0.21 vs 0.36, w0=−1 | Ωm | 2.512 | 19.87 | 443,621 | 14.05 % | 0.179 |
| Ωm 0.21 vs 0.36, w0=−1.4 | Ωm | 2.754 | 18.85 | 206,962 | 9.87 % | 0.279 |

ΔD/D = linear growth difference from z=47 to z=0 (cosmoprimo/CAMB).

- **The two dark-energy runs agree to 7 % across a 35× range in displacement.** Along
  w0/wa, |Δr| is simply proportional to the growth difference.
- **The Ωm runs sit 1.8–2.8× above that line** and disagree with each other: growth
  differences 14.1 % vs 9.9 %, yet medians 2.51 vs 2.75. There, displacement tracks
  **ΔΩm itself, not ΔD**.

**Why:** dark energy only rescales the growth *amplitude*, so a growth normalisation
absorbs it. Ωm also changes the *shape* of the initial power spectrum (k_eq ∝ Ωm·h) and
the IC amplitude (σ8 varies 8.5 % across Ωm). The field is reshaped, not rescaled.

⇒ **Prediction to test:** growth normalisation should collapse the dark-energy direction
to ~0 and leave a coherent Ωm residual.

### ⚠️ Correction to the previous report
I previously said "the w0/wa axis is safe". That rested on one pair which turns out to be
**nearly degenerate**: Δw0=+0.2 and Δwa=−0.8 cross at a≈0.75 in CPL, so the two expansion
histories almost coincide (ΔD/D = 0.28 %). Vary w0 alone by 0.4 and the median jumps 35×.
The axis label was never the point — the growth difference is.

---

## 4. Questions raised last meeting

| question | answer |
|----------|--------|
| **PBC bug? the movers piled up at 0** | **No bug.** That plot came from an 8-subfile smoke test, which can only see z = 0–34 (3 % of the box). Across the six full-box scans the top movers span z = 17–966 and x, y = 3–1014. **[movers_compare.png]** |
| Full histogram, not just percentiles | `--stats-out` now dumps the complete \|Δr\| histogram |
| Are the first/last slabs behaving? | per-slab median + edge/interior ratio: 1.030 and 0.872, within the natural slab-to-slab spread (0.94–1.20, 1.41–1.74) |
| Is the motion directional? | per-axis rms recorded; spread across x/y/z is 6.1 % and 11.3 % — close to isotropic |
| Are the clustered regions special? | movers repeat at the *same* halos across scans — (499.7, 703.3, 416.8) holds 9 of the top 20 in one scan, (685.3, 1013.3, 263.9) holds 6 in another. Whole structures shift together; individual particles do not scatter |
| Distribute over CPUs / MPI | done — §5 |

---

## 5. Parallelisation

Slabs are independent, so A's 250 subfiles split across workers with no communication
needed; MPI would add dependencies without benefit. Each worker writes its own `.npz` and
a merge step sums them — verified **bit-identical** to a single-process run (histogram,
threshold counts, per-axis sums, top-20 and unmatched all match exactly).

| | machine | processes | wall time |
|--|---------|-----------|-----------|
| before | login node | 1 | **13 h 20 min** (01:43 → 15:03) |
| after | compute node via SLURM | 16 | **48 min** (03:37:25 → 04:25:36) |

**≈17×.** Not a controlled benchmark — different cosmology pairs and different machines — but
the practical effect is a pair per hour instead of per night, and the work is off the login
node. Worker CPU sits at ~47 %, so the bottleneck is now GPFS I/O rather than cores;
adding workers past ~16 would gain little.

---

## 6. Interpolation methods (mock validation so far)
**[methods_results.png]** 48³ particles × 36 mock cosmologies, median position error [cMpc/h]:

| | linear | quadratic | RBF | GP |
|--|--------|-----------|-----|-----|
| off-grid | 0.59 | 0.23 | **0.072** | 0.27 |
| leave-one-out | 0.031 | 0.080 | **0.0008** | 0.014 |

RBF beats linear by 8–40×; GP is competitive and also returns an uncertainty (0.02–0.04),
useful for flagging sparse regions of the grid. Quadratic helps off-grid but hurts at a
held-out node — a global fit is not node-exact. All standard libraries (scipy, GPyTorch);
the GPU path tiles 2048³ with CuPy.

---

## 7. Data notes
- **`SyncINITIAL` is the grid to use, but its steps are NOT identical everywhere.**
  Om0.26_-1.2_-0.8 has 11 (00001 00281 00345 00441 00601 00921 01090 01241 01561 01706
  01881); Om0.21_-1_0 has 9, missing 00281/00441/01706 and adding 01631. The usable
  common ladder is the intersection — 8 steps so far: **00001 00345 00601 00921 01090
  01241 01561 01881**, with 01881 = z=0. Worth enumerating across all cosmologies
  before fixing the grid. (`INITIAL` is worse: cosmology-dependent ad-hoc dumps.)
- **Ωm = 0.31 has no `SyncINITIAL.01881`**, which is why the ΔΩm = 0.10 rung had to start
  from 0.26 instead of 0.21. Worth checking whether it exists elsewhere.
- Cosmologies are split across `/gpfs` and `/multiverse`, and a directory can exist in one
  path as an empty shell (logs only) — **count particle files, don't trust `ls -d`**.
- The `s09` variant uses a different σ8 convention → exclude from the grid.

---

## 8. Next
1. **The actual interpolation, on real data**: hold out Ωm = 0.26 and predict it from 0.21
   and 0.36 (w0=−1, wa=0); compare per-particle and via P(k). The premise work supports it.
2. **Test the growth-normalisation prediction** — it should collapse the dark-energy
   direction and leave an Ωm residual.
3. **More (w0, wa) pairs** — the proportionality currently rests on two points.
4. Settle coherence properly with a per-particle **dot product** of the two displacement
   vectors.

## ⚠️ Caveats to state plainly
- The dark-energy proportionality rests on **2 points**.
- The ΔΩm = 0.10 rung starts from 0.26, the others from 0.21, so the ladder is not
  perfectly controlled — the response could depend on absolute Ωm, not only on ΔΩm. The
  coherence test is unaffected, since it only uses that 0.21→0.36 is the composition of
  the other two.
- Coherence is computed from medians, which is indicative rather than exact (see Next 4).
- All interpolation-accuracy numbers are still **mock**; real-data numbers do not exist yet.
