"""Shared-IC result figure: how far the same particle moves between cosmologies.

Plots the survival function -- fraction of particles with |dr| greater than x --
for the two smoke scans, which is the cleanest way to show the directionality:
changing (w0, wa) barely moves particles, changing Omega_m moves them a lot.

Numbers are the measured percentiles + threshold counts from verify_shared_ic.py
(z=0, SyncINITIAL.01881, 8 subfiles each).

Run with a matplotlib python:  /opt/anaconda3/bin/python3 make_dr_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "axes.unicode_minus": False})

SPACING = 0.5           # mean interparticle spacing, cMpc/h
HONG = 10.0             # Dr. Hong's criterion

# --- measured: (|dr|, fraction of particles exceeding it) ---
# percentiles give (value, 1 - p); threshold counts give (value, count/N) directly.
SAME = {   # Om 0.26, (-1.2,-0.8) vs (-1.0,-1.6);  N = 274,526,550
    "label": r"same $\Omega_m$ = 0.26  (only $w_0, w_a$ differ)",
    "color": "#2c7fb8",
    "dr":   [0.0316, 0.0575, 0.1000, 0.1738, 0.3981, 0.5,      1.0,      2.0],
    "frac": [0.50,   0.10,   0.01,   1e-3,   1e-4,   4.648e-5, 4.222e-6, 8.74e-8],
    "max": 2.9230,
}
DIFF = {   # Om 0.21 vs 0.36, both w0=-1, wa=0;  N = 263,412,363
    "label": r"different $\Omega_m$: 0.21 vs 0.36  (same $w_0, w_a$)",
    "color": "#d95f0e",
    "dr":   [2.5119, 4.3652, 6.3096, 7.9433, 9.5499, 0.5,     1.0,     2.0,     5.0,      10.0],
    "frac": [0.50,   0.10,   0.01,   1e-3,   1e-4,   0.99139, 0.93590, 0.64907, 0.040700, 3.28e-5],
    "max": 13.9613,
}

fig, ax = plt.subplots(figsize=(9, 6.4))

for d in (SAME, DIFF):
    o = np.argsort(d["dr"])
    x, y = np.array(d["dr"])[o], np.array(d["frac"])[o]
    ax.loglog(x, y, "o-", color=d["color"], lw=2.2, ms=6, label=d["label"], zorder=3)
    ax.plot([d["max"]], [1e-9], "*", color=d["color"], ms=16, mec="black", mew=0.6, zorder=4)
    ax.annotate("max %.2f" % d["max"], (d["max"], 1e-9), textcoords="offset points",
                xytext=(4, 8), fontsize=9, color=d["color"])

ax.axvline(SPACING, color="0.5", ls=":", lw=1.4, zorder=1)
ax.text(SPACING * 1.08, 3e-8, "interparticle\nspacing 0.5", fontsize=9, color="0.35")
ax.axvline(HONG, color="crimson", ls="--", lw=1.6, zorder=1)
ax.text(HONG * 1.08, 3e-3, "Hong's 10 cMpc/h\ncriterion", fontsize=9.5, color="crimson")

ax.set_xlabel(r"$|\Delta r|$  between the two cosmologies  [cMpc/h]")
ax.set_ylabel(r"fraction of particles with $|\Delta r|$ greater than $x$")
ax.set_xlim(0.02, 30)
ax.set_ylim(5e-10, 2)
ax.grid(alpha=0.25, which="both")
ax.legend(loc="lower left", fontsize=10.5, framealpha=0.95)
ax.set_title("Same Lagrangian particle, two cosmologies, $z=0$\n"
             "shared-IC premise holds along $w_0/w_a$, weakens along $\\Omega_m$",
             fontsize=13)

out = "/Users/hyunohyeo/Desktop/shared_ic_dr.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("same Om : median %.4f, max %.2f" % (SAME["dr"][0], SAME["max"]))
print("diff Om : median %.4f, max %.2f  (%.0fx larger median)"
      % (DIFF["dr"][0], DIFF["max"], DIFF["dr"][0] / SAME["dr"][0]))
