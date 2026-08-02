# Meeting notes — 2026-07-27

## 🎯 One-line takeaway
> Verified the shared-IC premise on real data and found it is **directional**:
> **along w0/wa particles barely move (interpolation easy); along Ωm they move a lot (hard).**
> → Ωm is the hard axis, and growth-factor normalization is essential there —
> shown from the data, not assumed.

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

### Result: opposite extremes depending on direction

| comparison | median | >5 Mpc | >10 Mpc | max | N | unmatched |
|------------|--------|--------|---------|-----|---|-----------|
| **same Ωm** (0.26, only w0/wa differ) — **FULL BOX** | **0.0302** | **4** | **0** | **5.52** | **8.59 B** | **0** |
| **different Ωm** (0.21 vs 0.36) — smoke, 8/250 | **2.51** | 4.1% | 8,640 | 13.96+ | 263 M | 4.2% |

- **The same-Ωm scan is complete and lossless.** 8,589,934,592 particles compared — exactly
  2048³, i.e. **every particle in the box**: none dropped at slab boundaries, none outside
  the search window. Median motion is **1/16.5 of the interparticle spacing**; only
  **4 particles in the entire box** exceed 5 cMpc/h and **none exceeds 10**. Even the single
  worst particle (5.52) is 1.8× under the criterion.
- **Different Ωm** (corner to corner): median is **5× the spacing** — **83× larger** than the
  same-Ωm median. The premise **weakens sharply along this axis**.

### No periodic-boundary bug
The 8-subfile smoke made the top movers look piled up near coordinate 0. That was its
truncated z-range (0–34 of 1024), not a wrap error: in the full scan the largest movers are
spread through the box — (685, 1013, 264), (910, 418, 71), (294, 821, 963), (529, 234, 946),
(78, 290, 60), (559, 220, 833) — with no edge preference.

### Why — the growth difference
Different Ωm ⇒ different amount of structure growth by z=0 ⇒ particles systematically
rearrange. Changing w0/wa is negligible by comparison, because dark energy only acts late.

### Sensitive locations are real and clustered
**[figure: movers_compare.png]** The biggest movers are not scattered — they **concentrate
on specific halos**. In the full same-Ωm scan, 6 of the top 20 sit within ~1 cMpc/h of
(685.3, 1013.3, 263.9), with two more pairs at (294, 821, 963) and (910, 418, 71).

This matches the expectation discussed last time: most particles stay put, while a few
dense spots are genuinely cosmology-sensitive.

---

## 3. What this means for the project (not fatal — it points the way)

1. **Interpolation happens between adjacent grid points.** The different-Ωm result above is
   corner-to-corner (ΔΩm = 0.15); the actual grid spacing is 0.05, so real motion should be
   substantially smaller. **Next: measure adjacent Ωm (0.21 ↔ 0.26).**
2. **Growth-factor normalization is essential along Ωm.** Most of the motion is the growth
   difference, so normalizing to a common growth amplitude should shrink |Δr| a lot
   (`cpl_growth.py`, cosmoprimo). This result turns that step from optional to required.
3. **Ωm = hard axis, w0/wa = easy axis** — this should shape the interpolation design
   (e.g. more care, or more training points, along Ωm).

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

## 6. Next (action items from this meeting)
1. **Interpolate varying Ωm only** — the hard axis, and the cleanest first test
2. **Does |Δr| scale smoothly with ΔΩm?** measure adjacent (0.05) and mid (0.10) spacings.
   Large motion is fine for interpolation *if* it is smooth; that is the real question.
3. **Different-Ωm full scan** with a wider window — true max and complete tail
4. **Parallelize**: slabs are independent, so this maps cleanly onto MPI/multi-process.
   Currently single-core, ~12 h per pair, and running on the **login node** — should move
   to SLURM.
5. Re-measure |Δr| **after growth-factor normalization** — how much does it shrink?
6. Are the clustered mover regions special (halo? void?), and is the **direction** of
   motion there systematic? Diagnostics now available via `--stats-out`.
7. Evaluate interpolation accuracy via the **P(k) ratio**

---

## Figures
1. **w0wa_grid.png** — the 50 cosmologies in (w0, wa), colored by Ωm
2. **shared_ic_dr.png** ★ — the key figure. Survival curves, same-Ωm vs different-Ωm.
   Read as "fraction of particles that moved more than x". Blue never reaches the
   10 cMpc/h line; orange crosses it.
3. **movers_compare.png** — where the biggest movers sit (note the colorbar scales differ
   by an order of magnitude between the two rows)
4. **methods_results.png** — mock accuracy of the four models
5. **param_space.png** — if the σ8-vs-Ωm point needs making
6. **slabs_compare.png** — two cosmologies look structurally similar

## ⚠️ Be honest about
- **Different-Ωm is still a smoke test (8 of 250 subfiles).** Its median is reliable over
  263 M particles, but the **max and tail are underestimates**, and its mover positions only
  cover z = 0–34 cMpc/h (3% of the box) — which is what made them look piled up at 0.
- **The matching window bounds |Δz|, not |Δr|** — slabs are cut along z, so large x/y motion
  is measured exactly and only |Δz| > w×4.3 goes unmatched. The same-Ωm full scan has
  **0 unmatched**, so its tail is complete; the different-Ωm smoke lost 4.2% and is not.
- **Different-Ωm is corner-to-corner** (ΔΩm = 0.15); actual interpolation is adjacent (0.05).
