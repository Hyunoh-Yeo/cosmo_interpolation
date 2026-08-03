# Meeting notes — 2026-07-27

## 🎯 One-line takeaway
> **The Ωm axis is interpolatable.** At the spacing we actually interpolate over
> (ΔΩm = 0.05), only **41 particles out of 8.59 billion** exceed 10 cMpc/h, and the
> displacement field is **coherent** — successive Ωm steps add up (94 % of the linear
> sum) instead of scattering (which would have given 1.93 instead of 2.51). The earlier
> alarming numbers were all corner-to-corner (ΔΩm = 0.15), a case interpolation never
> has to handle.
>
> Underlying mechanism, from six full-box scans: displacement is **proportional to the
> linear growth difference for dark energy** (two runs agree to 7 % across a 35× range),
> while Ωm moves particles 1.8–2.8× further than that law — because Ωm reshapes the
> initial power spectrum rather than only rescaling it.

---

## 1. Parameter space — what we have, what we predict
**[figure: w0wa_grid.png]**

- 50 cosmologies scattered over (w0, wa) as an **irregular grid** (Ωm = 0.21/0.26/0.31/0.36,
  11–13 points each) → not a regular grid → needs **scattered-data interpolation** (RBF/GP).
- Most (w0, wa) positions carry all four Ωm values, so the grid has clean Ωm "columns".
- **[param_space.png]** the IC amplitude (σ8 at z=47) is set by **Ωm alone** — (w0, wa) shift
  it by only ~0.1%. By z=47 dark energy has not acted yet, so all (w0, wa) signal lives in
  the post-z=47 growth.
- The **s9 variant uses a different σ8 convention → excluded from the grid.**

---

## 2. Shared-IC verification — is the same particle really in ~the same place? ★KEY★
**[figure: shared_ic_dr.png]**

**Method**: match the same Lagrangian ID (`indx`) between two cosmologies at z=0 and measure
the position difference |Δr|. Scan is unbiased — particles are matched by identity across
neighbouring z-slabs, so boundary-crossers (the biggest movers) are not silently dropped,
which is what the earlier single-subfile measurement did wrong.

### The decisive test: the ΔΩm ladder
**[figure: om_ladder.png]** Three full-box scans at w0=−1, wa=0:

| ΔΩm | pair | median | max | >10 cMpc/h |
|-----|------|--------|-----|------------|
| **0.05** (the real grid spacing) | 0.21→0.26 | **1.097** | 12.65 | **41** (4.8e-9) |
| 0.10 | 0.26→0.36 | 1.585 | 19.03 | 129,853 |
| 0.15 (corner to corner) | 0.21→0.36 | 2.512 | 19.87 | 443,621 |

1. **At ΔΩm = 0.05 only 41 particles in 8.59 G exceed 10 cMpc/h.** Every alarming number
   so far came from the corner-to-corner case, which interpolation never faces.
2. **The steps add, they do not scatter.** 0.15 is the composition of the other two:
   uncorrelated steps would give the quadrature sum 1.93, fully coherent steps the linear
   sum 2.68 — measured **2.51, i.e. 94 % of the linear sum.** The displacement field is
   coherent, which is precisely what makes it fittable.
3. **Mildly sublinear, no saturation**: |Δr| ∝ ΔΩm^0.75.

⇒ Ωm is hard in *magnitude* but well-behaved in *structure*. Large motion was never the
problem; unpredictable motion would have been, and it is not unpredictable.

### The mechanism — six complete-box scans
**[figure: dr_scaling.png]** All four are full-box (8.59 G particles each, `unmatched` ≤ 35,
i.e. 4e-9 of the box), so every tail is complete.

| scan | what differs | median | max | >10 cMpc/h | ΔD/D |
|------|--------------|--------|-----|------------|------|
| w0,wa: (−1.2,−0.8) vs (−1.0,−1.6), Ωm 0.26 | dark energy | **0.030** | 5.52 | **0** | **−0.28%** |
| w0 only: −1.0 vs −1.4, Ωm 0.21 | dark energy | **1.047** | 9.96 | **0** | 10.39% |
| Ωm: 0.21 vs 0.36, w0=−1 | **Ωm** | **2.512** | 19.87 | 443,621 | 14.05% |
| Ωm: 0.21 vs 0.36, w0=−1.4 | **Ωm** | **2.754** | 18.85 | 206,962 | 9.87% |

ΔD/D = difference in linear growth from z=47 to z=0 (cosmoprimo/CAMB).

**⚠️ Correction to what I said last time.** The first pair is **nearly degenerate**: its
Δw0=+0.2, Δwa=−0.8 cross at a≈0.75 in CPL, so the two expansion histories almost coincide
(ΔD/D = 0.28%). That, not "dark energy is harmless", is why its median is 0.030. Change w0
alone by 0.4 and the median jumps **35×**. So the axis labels matter less than the growth
difference.

**The real structure — dark energy follows a clean law, Ωm does not:**

| | median \|Δr\| per 1% of ΔD/D |
|--|---------------------------|
| w0,wa (degenerate pair) | **0.108** |
| w0 only | **0.101** |
| Ωm (w0=−1) | 0.179 |
| Ωm (w0=−1.4) | 0.279 |

- The two dark-energy runs agree to **7% across a 35× range in displacement** — along
  w0/wa, displacement is simply **proportional to the growth difference**.
- The Ωm runs sit **1.8–2.8× above** that line, and disagree with each other: their growth
  differences are 14.1% vs 9.9% yet their medians are nearly equal (2.51 vs 2.75). Along Ωm
  the displacement tracks **ΔΩm itself, not ΔD**.

**Why**: dark energy only rescales the growth amplitude, so growth normalization absorbs it
entirely. Ωm additionally changes the **shape** of the initial power spectrum (k_eq ∝ Ωm·h)
and the IC amplitude (σ8 spread 8.5% across Ωm). The field is not rescaled — it is reshaped.

⇒ **Prediction: growth normalization should fully fix the w0/wa direction and only partly
the Ωm direction.** That is the next thing to measure.

### No periodic-boundary bug
The 8-subfile smoke made the top movers look piled up near coordinate 0. That was its
truncated z-range (0–34 of 1024), not a wrap error: in the full scan the largest movers are
spread through the box — (685, 1013, 264), (910, 418, 71), (294, 821, 963), (529, 234, 946),
(78, 290, 60), (559, 220, 833) — with no edge preference.

### Sensitive locations are real and clustered
**[figure: movers_compare.png]** The biggest movers are not scattered — they **concentrate
on specific halos**. In the full same-Ωm scan, 6 of the top 20 sit within ~1 cMpc/h of
(685.3, 1013.3, 263.9), with two more pairs at (294, 821, 963) and (910, 418, 71).

This matches the expectation discussed last time: most particles stay put, while a few
dense spots are genuinely cosmology-sensitive.

---

## 3. What this means for the project

1. **The premise holds where it needs to.** At the real grid spacing the criterion is met
   by ~8 orders of magnitude (41 particles in 8.59 G), and the displacement field is
   coherent rather than chaotic. The verification phase can close.
2. **Growth normalization is the right pre-processing step**, and we now know exactly what
   it will and will not do: it should absorb the dark-energy direction entirely (|Δr| ∝ ΔD
   there) and leave a coherent Ωm residual from the power-spectrum shape change.
3. **Ωm is the axis that carries real signal** — w0/wa pairs can even be near-degenerate.
   So the interpolator should be given room along Ωm, and the (w0, wa) grid can be judged
   by growth difference rather than by parameter distance.

---

## 4. Interpolation methods (mock validation)
**[figure: methods_results.png]**

4 models on mock data (48³ particles × 36 cosmologies), median per-particle position
error [cMpc/h]:

| | linear | quadratic | RBF | GP |
|--|--------|-----------|-----|-----|
| off-grid | 0.59 | 0.23 | **0.072** | 0.27 |
| leave-one-out | 0.031 | 0.080 | **0.0008** | 0.014 |

- **RBF beats linear by 8–40×.** GP is competitive and additionally returns an uncertainty
  (0.02–0.04), useful for flagging where the grid is too sparse.
- Quadratic helps off-grid but *hurts* at a held-out node — a global fit is not node-exact.
- All standard libraries (scipy, GPyTorch); the GPU path tiles 2048³ with CuPy.

---

## 5. Done so far
- ✅ GOTPM format decoded, reader validated on real files
- ✅ Subfile layout verified — 250 slabs, ordered in z, gap 0.000, overlap 0.000
- ✅ Shared-IC directionality found (section 2)
- ✅ First real-data P(k) via pypower (shot noise matches V/N exactly)
- ✅ Parameter-space analysis

## 6. Next
1. **Interpolate, on real data.** The premise work is done and the answer is favourable,
   so the next thing is the actual deliverable: hold out Ωm = 0.26 and predict it from
   0.21 and 0.36 (w0=−1, wa=0), then compare per-particle and via P(k).
2. **More (w0, wa) pairs** to confirm the dark-energy proportionality with >2 points.
3. **Test the prediction**: re-measure |Δr| after growth normalization. It should collapse
   the dark-energy cases to ~0 and leave a residual for Ωm. If that holds, the residual is
   the shape term and can be modelled separately.
4. **Interpolate varying Ωm only** — the hard axis, the cleanest first real-data test.
5. Are the clustered mover regions halos or voids, and is the **direction** of motion there
   systematic? `--stats-out` now records per-axis and per-slab statistics.
6. Evaluate interpolation accuracy via the **P(k) ratio**.

✅ **Parallelization is done** — `--sub-range` + `merge_stats.py` + `job_verify_ic.sh`.
Slabs are independent, so A's 250 subfiles split across 16 workers on one node
(64 cores, 500 GB); the merge is bit-identical to a serial run (verified). ~12 h → ~1 h,
and it moves the work off the login node onto SLURM.

---

## Figures
1. **om_ladder.png** ★★ — the decisive one. (a) |Δr| vs ΔΩm, mildly sublinear.
   (b) coherence: measured step = 94% of the linear sum, not the quadrature sum.
   (c) at ΔΩm = 0.05, only 41 particles in 8.59 G break 10 cMpc/h.
2. **dr_scaling.png** ★ — the mechanism. (a) displacement vs growth difference: the two
   dark-energy runs fall on one line through the origin, the Ωm runs sit 1.7–2.7× above it.
   (b) four scans, median and max, against the 10 cMpc/h reference.
3. **w0wa_grid.png** — the 50 cosmologies in (w0, wa), colored by Ωm
3. **shared_ic_dr.png** — survival curves (fraction moving more than x) for the degenerate
   w0/wa pair vs the Ωm pair; the two are cleanly separated.
3. **movers_compare.png** — where the biggest movers sit (note the colorbar scales differ
   by an order of magnitude between the two rows)
4. **methods_results.png** — mock accuracy of the four models
5. **param_space.png** — if the σ8-vs-Ωm point needs making
6. **slabs_compare.png** — two cosmologies look structurally similar

## ⚠️ Be honest about
- **The dark-energy proportionality rests on 2 points.** Striking, but it needs more
  (w0, wa) pairs before being called a law.
- **The ΔΩm = 0.10 rung starts from 0.26, the others from 0.21.** So the ladder is not
  perfectly controlled — the response could depend on absolute Ωm, not only on ΔΩm.
  (Ωm = 0.31 has no SyncINITIAL.01881, which is why.) The coherence test is unaffected,
  since it only uses the fact that 0.21→0.36 is the composition of the two steps.
- **Coherence is measured on medians**, which is indicative rather than exact; a
  per-particle dot product of the two displacement vectors would settle it properly.
- **The matching window bounds |Δz|, not |Δr|** — slabs are cut along z, so large x/y motion
  is measured exactly and only |Δz| > w×4.3 goes unmatched. Here that cost at most 35
  particles out of 8.59 G, so all four tails are effectively complete.
- The growth factors come from CAMB via cosmoprimo, cross-checked earlier against an
  in-house ODE to 0.03%.
