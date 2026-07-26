"""Parameter-space figure for the Multiverse suite.

Data = the 50 `camb.z=47_<Om>_<w0>_<wa>.ascii` headers on grammar (Om, w0, wa and
sigma8 at z=47). Shows (a) which (w0,wa) points exist for each Omega_m, and
(b) that the IC amplitude sigma8(z=47) is essentially a function of Omega_m alone.

Run with a matplotlib python:  /opt/anaconda3/bin/python3 make_param_space_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({"font.size": 11, "figure.dpi": 140, "axes.unicode_minus": False})

# (Om, w0, wa, sigma8@z=47) — read from the CAMB headers on grammar
DATA = [
    (0.21, -1.40, 0.00, 0.0215339), (0.21, -1.20, -0.80, 0.0215339),
    (0.21, -1.20, 0.80, 0.0215032), (0.21, -1.00, -1.60, 0.0215339),
    (0.21, -0.90, 0.40, 0.0215083), (0.21, -0.80, -2.40, 0.0215339),
    (0.21, -0.60, -1.60, 0.0215339), (0.21, -0.60, 0.00, 0.0215120),
    (0.21, -0.40, -2.40, 0.0215339), (0.21, -0.40, -0.80, 0.0215339),
    (0.21, -0.20, -1.60, 0.0215340),
    (0.26, -1.40, 0.00, 0.0223218), (0.26, -1.20, -0.80, 0.0223218),
    (0.26, -1.20, 0.80, 0.0222975), (0.26, -1.00, -1.60, 0.0223218),
    (0.26, -1.00, 0.00, 0.0253129),                      # <- s9 variant (different sigma8 convention)
    (0.26, -0.90, 0.40, 0.0223016), (0.26, -0.80, -2.40, 0.0223218),
    (0.26, -0.80, -0.80, 0.0223218), (0.26, -0.60, -1.60, 0.0223219),
    (0.26, -0.60, 0.00, 0.0223046), (0.26, -0.40, -2.40, 0.0223219),
    (0.26, -0.40, -0.80, 0.0223220), (0.26, -0.20, -1.60, 0.0223220),
    (0.31, -1.40, -1.60, 0.0229076), (0.31, -1.40, 0.00, 0.0229077),
    (0.31, -1.20, -2.40, 0.0229076), (0.31, -1.20, -0.80, 0.0229077),
    (0.31, -1.20, 0.80, 0.0228884), (0.31, -1.00, -1.60, 0.0229077),
    (0.31, -0.90, 0.40, 0.0228921), (0.31, -0.80, -2.40, 0.0229078),
    (0.31, -0.80, -0.80, 0.0229082), (0.31, -0.60, -1.60, 0.0229082),
    (0.31, -0.40, -2.40, 0.0229082), (0.31, -0.40, -0.80, 0.0229089),
    (0.31, -0.20, -1.60, 0.0229089),
    (0.36, -1.40, -1.60, 0.0233486), (0.36, -1.40, 0.00, 0.0233486),
    (0.36, -1.20, -2.40, 0.0233486), (0.36, -1.20, -0.80, 0.0233456),
    (0.36, -1.20, 0.80, 0.0233310), (0.36, -1.00, -1.60, 0.0233486),
    (0.36, -0.90, 0.40, 0.0233417), (0.36, -0.80, -2.40, 0.0233535),
    (0.36, -0.80, -0.80, 0.0233570), (0.36, -0.60, -1.60, 0.0233497),
    (0.36, -0.40, -2.40, 0.0233570), (0.36, -0.40, -0.80, 0.0233513),
    (0.36, -0.20, -1.60, 0.0233498),
]
S9 = (0.26, -1.00, 0.00)                      # excluded: different sigma8 normalisation
TARGET = (-1.10, -1.20)                       # example (w0,wa) we would predict

arr = np.array(DATA)
OMS = [0.21, 0.26, 0.31, 0.36]
COL = {0.21: "#4c72b0", 0.26: "#dd8452", 0.31: "#55a868", 0.36: "#c44e52"}

fig = plt.figure(figsize=(14.5, 8.4))
gs = GridSpec(2, 4, height_ratios=[1.35, 0.95], hspace=0.62, wspace=0.12)

# --- top row: (w0, wa) points for each Omega_m ---
for i, om in enumerate(OMS):
    ax = fig.add_subplot(gs[0, i])
    sel = arr[np.isclose(arr[:, 0], om)]
    is_s9 = np.isclose(sel[:, 1], S9[1]) & np.isclose(sel[:, 2], S9[2]) & np.isclose(om, S9[0])
    ax.scatter(sel[~is_s9, 1], sel[~is_s9, 2], s=70, color=COL[om],
               edgecolor="black", linewidth=0.6, zorder=3, label="simulated")
    if is_s9.any():
        ax.scatter(sel[is_s9, 1], sel[is_s9, 2], s=130, marker="x", color="red",
                   linewidth=2.4, zorder=4, label="s9 variant (exclude)")
    ax.scatter(*TARGET, s=180, marker="*", color="gold", edgecolor="black",
               linewidth=0.8, zorder=5, label="example target")
    ax.set_title(r"$\Omega_m$ = %.2f   (n=%d)" % (om, len(sel)), fontsize=12)
    ax.set_xlabel(r"$w_0$")
    if i == 0:
        ax.set_ylabel(r"$w_a$")
    else:
        ax.set_yticklabels([])
    ax.set_xlim(-1.55, -0.05); ax.set_ylim(-2.75, 1.15)
    ax.grid(alpha=0.25)
    if i == 3:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)

# --- bottom: sigma8(z=47) vs Omega_m ---
ax = fig.add_subplot(gs[1, :])
for om in OMS:
    sel = arr[np.isclose(arr[:, 0], om)]
    is_s9 = np.isclose(sel[:, 1], S9[1]) & np.isclose(sel[:, 2], S9[2]) & np.isclose(om, S9[0])
    ax.scatter(np.full((~is_s9).sum(), om), sel[~is_s9, 3], s=55, color=COL[om],
               edgecolor="black", linewidth=0.5, zorder=3)
    if is_s9.any():
        ax.scatter([om], sel[is_s9, 3], s=130, marker="x", color="red", linewidth=2.4, zorder=4)
        ax.annotate("s9 variant — different $\\sigma_8$ convention → exclude",
                    (om, sel[is_s9, 3][0]), textcoords="offset points", xytext=(16, -3),
                    fontsize=9.5, color="red", va="center")
    spread = 100 * (sel[~is_s9, 3].max() - sel[~is_s9, 3].min()) / sel[~is_s9, 3].mean()
    ax.annotate("%d sims\nspread %.2f%%" % ((~is_s9).sum(), spread),
                (om, sel[~is_s9, 3].mean()), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=9, color=COL[om])
ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$\sigma_8$ at $z=47$" "\n(IC amplitude)")
ax.set_title(r"IC amplitude is set by $\Omega_m$ alone — $(w_0,w_a)$ move it by only ~0.1%"
             "   (by $z=47$ dark energy has not acted yet)", fontsize=11.5, pad=10)
ax.set_xticks(OMS); ax.set_xlim(0.185, 0.395); ax.grid(alpha=0.25)

fig.suptitle("Multiverse parameter space — 50 simulated cosmologies "
             "(irregular grid → needs scattered-data interpolation)", fontsize=13.5)
fig.savefig("/Users/hyunohyeo/Desktop/param_space.png", bbox_inches="tight")
print("saved /Users/hyunohyeo/Desktop/param_space.png")

# quick text summary
print("\ncosmologies per Omega_m:", {om: int(np.isclose(arr[:, 0], om).sum()) for om in OMS})
w0s = sorted(set(np.round(arr[:, 1], 2))); was = sorted(set(np.round(arr[:, 2], 2)))
print("w0 values:", w0s)
print("wa values:", was)
print("full grid would be %d x %d = %d per Om; actual is 11-13 -> irregular"
      % (len(w0s), len(was), len(w0s) * len(was)))
