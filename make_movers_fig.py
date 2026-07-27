"""Where the biggest movers sit, same-Omega_m vs different-Omega_m.

Same layout as plot_movers.py (three projections, colored by |dr|), but built from
the top-20 lists the two smoke scans printed, so the two cases can be compared
side by side before a --save-movers run exists.

CAVEAT: 20 particles each, and the smoke scans only read 8 subfiles, so z is
confined to 0-34 cMpc/h. This shows the clustering, not the full spatial map --
for that, re-run verify_shared_ic.py with --save-movers and use plot_movers.py.

Run with a matplotlib python:  /opt/anaconda3/bin/python3 make_movers_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.unicode_minus": False})

BOX = 1024.0
ZMAX = 34.0          # the 8-subfile smoke reach

# (|dr|, x, y, z) -- top 20 from each scan, z=0, SyncINITIAL.01881
SAME = np.array([        # Om 0.26: (-1.2,-0.8) vs (-1.0,-1.6)
    (2.9230, 205.38, 524.54, 11.37), (2.7621, 698.08, 609.97, 16.60),
    (2.7107,  53.46, 649.65, 32.95), (2.5326, 183.40,  94.00, 23.05),
    (2.4898,  53.03, 649.52, 32.79), (2.3347,  53.25, 648.52, 32.18),
    (2.3222,  53.49, 648.55, 32.42), (2.2884, 483.99, 816.83, 25.98),
    (2.2148,  53.38, 649.11, 32.95), (2.2049, 949.17, 304.01, 22.93),
    (2.2028, 185.12,  95.96, 25.74), (2.1920, 484.01, 816.91, 26.10),
    (2.1901, 475.42, 821.48, 22.48), (2.1587,  46.93, 707.35, 15.47),
    (2.1391, 920.75, 573.16, 28.62), (2.1217,  53.58, 649.24, 33.25),
    (2.0865, 484.06, 816.90, 26.04), (2.0844, 721.20, 327.59,  2.69),
    (2.0684, 484.76, 816.63, 24.31), (2.0430, 793.59, 768.14, 28.58),
])
DIFF = np.array([        # Om 0.21 vs 0.36, both w0=-1, wa=0
    (13.9613,   3.39, 981.17,  7.64), (13.8113, 662.96, 232.06,  7.22),
    (13.7467,   3.54, 983.62,  9.63), (13.6887,   0.56, 984.69, 12.28),
    (13.4962, 662.94, 231.47,  7.55), (13.1512,  45.69, 708.07, 15.63),
    (13.1497,  15.18, 944.34,  1.05), (13.0705, 190.64, 100.64, 26.73),
    (12.9751,   3.77, 981.63,  9.22), (12.9736,  15.17, 944.32,  0.95),
    (12.8875,   2.89, 982.82, 10.18), (12.8385, 664.66, 232.17,  7.48),
    (12.8358, 646.37, 984.13, 16.27), (12.8325, 610.13, 448.29, 33.93),
    (12.7981,   3.68, 983.40,  9.62), (12.7884,   3.13, 981.01,  8.87),
    (12.7813, 189.24,  93.60, 28.40), (12.7781, 936.28, 602.07,  3.21),
    (12.7751, 189.22,  93.49, 28.34), (12.7306,   2.83, 982.52, 10.12),
])

CASES = [
    (SAME, r"same $\Omega_m$ = 0.26  (only $w_0,w_a$ differ)", "Blues_r"),
    (DIFF, r"different $\Omega_m$: 0.21 vs 0.36", "autumn_r"),
]
PROJ = [("x", "y", 1, 2, BOX), ("x", "z", 1, 3, ZMAX), ("y", "z", 2, 3, ZMAX)]

fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
for row, (d, title, cmap) in enumerate(CASES):
    o = np.argsort(d[:, 0])
    d = d[o]                                   # biggest on top
    for col, (la, lb, ia, ib, hi) in enumerate(PROJ):
        ax = axes[row, col]
        sc = ax.scatter(d[:, ia], d[:, ib], c=d[:, 0], s=90, cmap=cmap,
                        edgecolor="black", linewidth=0.5,
                        vmin=d[:, 0].min(), vmax=d[:, 0].max())
        ax.set_xlabel("%s [cMpc/h]" % la); ax.set_ylabel("%s [cMpc/h]" % lb)
        ax.set_xlim(0, BOX if ia != 3 else hi)
        ax.set_ylim(0, hi)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=12, loc="left")
        if lb == "z":
            ax.text(0.98, 0.94, "z limited to 0-%d\n(8-subfile smoke)" % ZMAX,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, color="0.4")
    fig.colorbar(sc, ax=axes[row, :], label=r"$|\Delta r|$ [cMpc/h]",
                 pad=0.01, fraction=0.02)

fig.suptitle("Largest movers (top 20 each): where the shared-IC premise is least accurate",
             fontsize=13.5, y=0.97)
out = "/Users/hyunohyeo/Desktop/movers_compare.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)

for d, name, _ in CASES:
    cell = 20.0
    key = np.floor(d[:, 1:] / cell).astype(int)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    top = cnt.max()
    print("%-46s |dr| %.2f-%.2f   most crowded 20 cMpc/h cell holds %d of 20"
          % (name.replace("$", "").replace("\\Omega_m", "Om"), d[:, 0].min(), d[:, 0].max(), top))
