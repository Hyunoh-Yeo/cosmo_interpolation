"""Movers, projections only: the 8-subfile smoke against the six full-box scans.

Three panels (x-y, x-z, y-z) and nothing else, so the one point is unmissable --
the smoke's movers sit in a thin z = 0-34 band because it read 8 of 250 subfiles,
while the full scans spread over the whole box.

Positions are the top-20 lists the reports printed, so no new run is needed.
Data lives in make_movers_fig.py; this only changes the layout.

Run with a matplotlib python:  /opt/anaconda3/bin/python3 make_movers_proj_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "figure.dpi": 150, "axes.unicode_minus": False})
BOX = 1024.0

# reuse the tables from the main figure script, without importing its plotting
_src = open("make_movers_fig.py").read().split("allfull =")[0]
for drop in ('matplotlib.use("Agg")', "import matplotlib.pyplot as plt", "import matplotlib",
             'plt.rcParams.update({"font.size": 11, "figure.dpi": 150, '
             '"axes.unicode_minus": False})'):
    _src = _src.replace(drop, "")
_ns = {}
exec(_src, _ns)
SMOKE, FULL = _ns["SMOKE"], _ns["FULL"]
allfull = np.concatenate(list(FULL.values()))

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
PROJ = [("x", "y", 1, 2), ("x", "z", 1, 3), ("y", "z", 2, 3)]
cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(FULL)))

for ax, (la, lb, ia, ib) in zip(axes, PROJ):
    for (name, d), c in zip(FULL.items(), cmap):
        ax.scatter(d[:, ia], d[:, ib], s=46, color=c, edgecolor="black", lw=0.4,
                   label=name if ia == 1 and ib == 2 else None)
    ax.scatter(SMOKE[:, ia], SMOKE[:, ib], s=64, marker="^", color="#d62728",
               edgecolor="black", lw=0.5, zorder=4,
               label="8-subfile smoke" if ia == 1 and ib == 2 else None)
    if lb == "z":                       # shade the only band the smoke could see
        ax.axhspan(0, 34, color="#d62728", alpha=0.12, zorder=0)
        ax.text(BOX * 0.97, 52, "all the smoke could see:\n8 of 250 subfiles = z 0-34",
                ha="right", fontsize=8.5, color="#a01f1f")
    ax.set_xlabel("%s [cMpc/h]" % la); ax.set_ylabel("%s [cMpc/h]" % lb)
    ax.set_xlim(0, BOX); ax.set_ylim(0, BOX)
    ax.set_aspect("equal"); ax.grid(alpha=0.22)
    ax.set_title("%s-%s" % (la, lb), fontsize=12.5)

axes[0].legend(fontsize=8, loc="lower left", framealpha=0.95, ncol=1)
fig.suptitle("Top-20 movers per scan: 6 full-box comparisons (250/250 subfiles) "
             "vs the 8-subfile smoke", fontsize=13, y=1.0)
out = "/Users/hyunohyeo/Desktop/movers_proj.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("smoke  z %.1f-%.1f  (n=%d)" % (SMOKE[:, 3].min(), SMOKE[:, 3].max(), len(SMOKE)))
print("full   z %.1f-%.1f  (n=%d from %d scans)"
      % (allfull[:, 3].min(), allfull[:, 3].max(), len(allfull), len(FULL)))
