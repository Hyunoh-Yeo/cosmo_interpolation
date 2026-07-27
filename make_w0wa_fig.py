"""Single (w0, wa) scatter of the whole Multiverse suite, colored by Omega_m.

One panel: x = w0, y = wa, every simulated cosmology as a point. Shows the
irregular sampling that motivates scattered-data interpolation. Points are jittered
in Omega_m only via color, so overlapping (w0,wa) across Omega_m stay visible.

Run with a matplotlib python:  /opt/anaconda3/bin/python3 make_w0wa_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "axes.unicode_minus": False})

# (Om, w0, wa) — the 50 CAMB headers on grammar (sigma8 dropped here)
DATA = [
    (0.21, -1.40, 0.00), (0.21, -1.20, -0.80), (0.21, -1.20, 0.80), (0.21, -1.00, -1.60),
    (0.21, -0.90, 0.40), (0.21, -0.80, -2.40), (0.21, -0.60, -1.60), (0.21, -0.60, 0.00),
    (0.21, -0.40, -2.40), (0.21, -0.40, -0.80), (0.21, -0.20, -1.60),
    (0.26, -1.40, 0.00), (0.26, -1.20, -0.80), (0.26, -1.20, 0.80), (0.26, -1.00, -1.60),
    (0.26, -1.00, 0.00), (0.26, -0.90, 0.40), (0.26, -0.80, -2.40), (0.26, -0.80, -0.80),
    (0.26, -0.60, -1.60), (0.26, -0.60, 0.00), (0.26, -0.40, -2.40), (0.26, -0.40, -0.80),
    (0.26, -0.20, -1.60),
    (0.31, -1.40, -1.60), (0.31, -1.40, 0.00), (0.31, -1.20, -2.40), (0.31, -1.20, -0.80),
    (0.31, -1.20, 0.80), (0.31, -1.00, -1.60), (0.31, -0.90, 0.40), (0.31, -0.80, -2.40),
    (0.31, -0.80, -0.80), (0.31, -0.60, -1.60), (0.31, -0.40, -2.40), (0.31, -0.40, -0.80),
    (0.31, -0.20, -1.60),
    (0.36, -1.40, -1.60), (0.36, -1.40, 0.00), (0.36, -1.20, -2.40), (0.36, -1.20, -0.80),
    (0.36, -1.20, 0.80), (0.36, -1.00, -1.60), (0.36, -0.90, 0.40), (0.36, -0.80, -2.40),
    (0.36, -0.80, -0.80), (0.36, -0.60, -1.60), (0.36, -0.40, -2.40), (0.36, -0.40, -0.80),
    (0.36, -0.20, -1.60),
]
arr = np.array(DATA)
OMS = [0.21, 0.26, 0.31, 0.36]
COL = {0.21: "#4c72b0", 0.26: "#dd8452", 0.31: "#55a868", 0.36: "#c44e52"}
# tiny per-Omega offset so points sharing (w0,wa) across Omega_m stay visible
DX = {0.21: -0.018, 0.26: -0.006, 0.31: 0.006, 0.36: 0.018}

fig, ax = plt.subplots(figsize=(8.4, 7.2))
for om in OMS:
    sel = arr[np.isclose(arr[:, 0], om)]
    ax.scatter(sel[:, 1] + DX[om], sel[:, 2], s=95, color=COL[om],
               edgecolor="black", linewidth=0.6, zorder=3,
               label=r"$\Omega_m$ = %.2f" % om)

ax.set_xlabel(r"$w_0$")
ax.set_ylabel(r"$w_a$")
ax.set_xlim(-1.55, -0.05)
ax.set_ylim(-2.75, 1.15)
ax.grid(alpha=0.25, zorder=0)
ax.legend(fontsize=11, loc="lower left", framealpha=0.95)

out = "/Users/hyunohyeo/Desktop/w0wa_grid.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("points:", len(arr), "  per Omega_m:",
      {om: int(np.isclose(arr[:, 0], om).sum()) for om in OMS})
