"""Does displacement scale smoothly with Omega_m? The decisive test for whether the
Omega_m axis can be interpolated.

Three complete-box scans at z=0 (8.59 G particles each, w0=-1, wa=0 throughout):
  dOm 0.05  0.21 -> 0.26
  dOm 0.10  0.26 -> 0.36
  dOm 0.15  0.21 -> 0.36
The third is the composition of the first two, which turns it into a coherence test:
if the displacement field were random between steps the direct measurement would sit
at the quadrature sum; if fully coherent, at the linear sum.

Run with a matplotlib python:  /opt/anaconda3/bin/python3 make_omladder_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "figure.dpi": 150, "axes.unicode_minus": False})

N = 8589934592
# dOm, label, median, max, count > 10 cMpc/h
RUNS = [(0.05, r"$0.21\to0.26$", 1.0965, 12.6546, 41),
        (0.10, r"$0.26\to0.36$", 1.5849, 19.0338, 129853),
        (0.15, r"$0.21\to0.36$", 2.5119, 19.8687, 443621)]
SPACING = 0.5

dom = np.array([r[0] for r in RUNS])
med = np.array([r[2] for r in RUNS])
cnt = np.array([r[4] for r in RUNS], float)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.2))

# ---- (a) median vs dOm, with the linear expectation ----
ax1.plot(dom, med, "o-", color="#d95f0e", lw=2, ms=11, mec="black", mew=0.7, zorder=3)
xs = np.linspace(0, 0.17, 50)
ax1.plot(xs, med[0] / dom[0] * xs, "--", color="0.5", lw=1.5,
         label="linear in $\\Delta\\Omega_m$\n(anchored at the first point)")
p = np.log(med[2] / med[0]) / np.log(dom[2] / dom[0])
ax1.plot(xs, med[0] * (xs / dom[0]) ** p, ":", color="#d95f0e", lw=1.8,
         label=r"fit: $|\Delta r| \propto \Delta\Omega_m^{%.2f}$" % p)
for d, lab, m, _, _ in RUNS:
    ax1.annotate(lab, (d, m), textcoords="offset points", xytext=(6, -14), fontsize=9.5)
ax1.axhline(SPACING, color="0.6", ls=":", lw=1.2)
ax1.text(0.002, SPACING * 1.1, "interparticle spacing", fontsize=8.5, color="0.4")
ax1.set_xlabel(r"$\Delta\Omega_m$"); ax1.set_ylabel(r"median $|\Delta r|$  [cMpc/h]")
ax1.set_xlim(0, 0.17); ax1.set_ylim(0, 3.6)
ax1.grid(alpha=0.25); ax1.legend(fontsize=9, loc="upper left")
ax1.set_title("(a) mildly sublinear, no saturation", fontsize=12)

# ---- (b) coherence: linear sum vs quadrature vs measured ----
a, b, c = med
lin, quad = a + b, float(np.hypot(a, b))
bars = [("$0.21\\to0.26$", a, "#4c72b0"), ("$0.26\\to0.36$", b, "#4c72b0"),
        ("quadrature\n(random)", quad, "#999999"), ("linear sum\n(coherent)", lin, "#999999"),
        ("MEASURED\n$0.21\\to0.36$", c, "#d95f0e")]
for i, (lab, v, col) in enumerate(bars):
    ax2.bar(i, v, 0.62, color=col, edgecolor="black", lw=0.6,
            hatch="//" if col == "#999999" else None)
    ax2.text(i, v + 0.05, "%.2f" % v, ha="center", fontsize=10)
ax2.axhline(quad, color="0.5", ls=":", lw=1.2)
ax2.axhline(lin, color="0.5", ls=":", lw=1.2)
ax2.set_xticks(range(5))
ax2.set_xticklabels([b[0] for b in bars], fontsize=8.8)
ax2.set_ylabel(r"median $|\Delta r|$  [cMpc/h]")
ax2.set_ylim(0, 3.2); ax2.grid(alpha=0.25, axis="y")
f = (c - quad) / (lin - quad)
ax2.set_title("(b) steps ADD, they do not scatter\n"
              "measured = %.0f%% of the linear sum" % (100 * c / lin), fontsize=12)

# ---- (c) the practical number: how many break the 10 cMpc/h criterion ----
ax3.bar(range(3), cnt / N, 0.55, color=["#2ca02c", "#ff7f0e", "#d62728"],
        edgecolor="black", lw=0.6)
for i, (k, d) in enumerate(zip(cnt, dom)):
    ax3.text(i, max(k / N, 1e-9) * 1.5, "%d\n(%.0e)" % (int(k), k / N),
             ha="center", fontsize=9.5)
ax3.set_yscale("log")
ax3.set_xticks(range(3))
ax3.set_xticklabels([r"$\Delta\Omega_m=%.2f$" % d for d in dom], fontsize=10)
ax3.set_ylabel(r"fraction of the box with $|\Delta r| > 10$ cMpc/h")
ax3.set_ylim(1e-9, 3e-4)
ax3.grid(alpha=0.25, axis="y", which="both")
ax3.set_title("(c) at the real grid spacing (0.05)\nonly 41 particles in 8.59 G exceed 10",
              fontsize=12)

fig.suptitle(r"Is the $\Omega_m$ direction interpolatable?  Three full-box scans at $z=0$",
             fontsize=13.5, y=1.01)
out = "/Users/hyunohyeo/Desktop/om_ladder.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("power-law index %.3f ; coherence %.0f%% of linear sum (%.0f%% of the way from random)"
      % (p, 100 * c / lin, 100 * f))
