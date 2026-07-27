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

| comparison | median | >5 Mpc | >10 Mpc | max | N |
|------------|--------|--------|---------|-----|---|
| **same Ωm** (0.26, only w0/wa differ) | 0.032 | 0 | **0** | 2.92 | 275 M |
| same Ωm, full scan (150/250 done) | — | — | **0 unmatched** | — | **5.15 B** |
| **different Ωm** (0.21 vs 0.36) | **2.51** | 4.1% | **8,640** | 13.96+ | 263 M |

- **Same Ωm**: particles move ~**1/16 of the interparticle spacing** (0.5 cMpc/h). Nothing
  exceeds 10 cMpc/h — and over 5.15 B particles in the full scan, nothing even exceeds
  12.9. The premise holds cleanly, with a ≥3.4× margin on the 10 cMpc/h reference.
- **Different Ωm** (corner to corner): median is **5× the spacing**, and 8,640 particles
  exceed 10 cMpc/h. The premise **weakens along this axis**.
- Median differs by **79×** between the two cases. That gap is the result.

### Why — the growth difference
Different Ωm ⇒ different amount of structure growth by z=0 ⇒ particles systematically
rearrange. Changing w0/wa is negligible by comparison, because dark energy only acts late.

### Sensitive locations are real and clustered
**[figure: movers_compare.png]** The biggest movers are not scattered — they **concentrate
on specific halos**:
- same Ωm: 6 of the top 20 sit within one 20 cMpc/h cell at ≈(53, 649, 33)
- different Ωm: 8 of the top 20 within one cell at ≈(3, 982, 9)

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

## 6. Next steps
1. **Adjacent-Ωm |Δr|** (0.21 ↔ 0.26) — the real interpolation difficulty
2. Different-Ωm full scan with a **wider window**, to get the true max and complete tail
3. Re-measure |Δr| **after growth-factor normalization** — how much does it shrink?
4. Apply the interpolators to real snapshots (SyncINITIAL grid, aligned by `indx`)
5. Evaluate interpolation accuracy via the **P(k) ratio**

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
- **Different-Ωm is a smoke test (8 of 250 subfiles).** Distribution centre (median, 90%,
  99%) is reliable over 263 M particles, but the **max and tail are underestimates**, and
  the mover positions only cover z = 0–34 cMpc/h (3% of the box).
- **Same-Ωm full scan is at 150/250**, still 0 unmatched over 5.15 B particles.
- **Different-Ωm is corner-to-corner** (ΔΩm = 0.15); actual interpolation is adjacent (0.05).
- **The matching window bounds |Δz|, not |Δr|** — slabs are cut along z, so large x/y motion
  is still measured exactly, and only |Δz| > 12.9 goes unmatched. 4.2% went unmatched in the
  different-Ωm smoke, so that tail is **incomplete**; the same-Ωm run has 0 unmatched and is
  complete. Fix is a wider window.
