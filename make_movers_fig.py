"""Why did the movers look piled up at coordinate 0? Because that plot came from an
8-subfile smoke test.

Left: the smoke scan (8 of 250 subfiles). It can only see z = 0-34 cMpc/h, so every
mover it reports sits in the bottom 3% of the box -- an artefact of the truncation,
not a periodic-boundary bug.
Right: the six complete-box scans (250/250 subfiles). Their top movers span the full
0-1024 in every axis.

Positions are the top-20 lists each scan printed, so this needs no new run.

Run with a matplotlib python:  /opt/anaconda3/bin/python3 make_movers_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.unicode_minus": False})
BOX = 1024.0

# --- 8-subfile smoke, Om 0.21 vs 0.36: the run that produced the "piled up at 0" look ---
SMOKE = np.array([
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

# --- the six complete-box scans: top-20 each, as printed in their reports ---
FULL = {
 "same $\\Omega_m$ 0.26, $w_0w_a$": np.array([
    (5.5218,685.25,1013.29,263.91),(5.4722,909.77,418.01,71.33),(5.1223,688.35,1009.62,261.97),
    (5.0556,294.19,820.64,963.47),(4.9395,910.36,417.68,70.89),(4.9199,685.41,1013.22,263.83),
    (4.6754,683.79,1013.95,266.38),(4.5361,529.59,234.42,946.12),(4.4071,78.79,290.41,60.24),
    (4.3931,685.26,1013.31,263.70),(4.3901,294.55,820.69,963.23),(4.2355,688.57,1010.39,261.87),
    (4.0848,559.06,220.55,833.30),(4.0837,685.82,1012.11,263.72),(3.9685,685.41,1013.29,264.09),
    (3.8627,685.40,1013.32,264.01),(3.8622,555.94,219.74,830.35),(3.8382,353.60,533.59,956.30),
    (3.8342,140.22,825.79,460.78),(3.8189,294.05,818.66,966.31)]),
 "$\\Omega_m$ 0.21, $w_0$ scan": np.array([
    (9.9579,588.61,640.28,237.01),(9.5198,588.80,640.15,237.02),(8.7303,726.40,767.04,934.00),
    (8.6737,727.09,768.08,933.96),(8.6610,725.94,766.97,934.04),(8.5385,770.41,815.41,933.79),
    (8.5027,726.85,767.68,933.82),(8.4692,771.25,815.49,934.39),(8.4403,727.40,768.19,934.14),
    (8.4375,726.70,767.52,934.12),(8.4304,727.39,768.43,934.13),(8.4043,727.08,768.27,934.39),
    (8.3999,727.37,768.22,933.83),(8.3823,727.01,768.10,934.38),(8.3755,726.38,766.83,933.87),
    (8.3741,727.30,768.30,933.80),(8.3709,726.53,768.05,933.87),(8.3692,727.34,768.21,933.98),
    (8.3641,726.76,767.97,934.12),(8.3554,727.28,768.36,933.64)]),
 "$\\Delta\\Omega_m$ 0.05  (0.21$\\to$0.26)": np.array([
    (12.6546,916.23,413.51,63.05),(11.4345,353.56,536.17,956.63),(11.1796,858.33,628.75,430.98),
    (11.1654,858.23,629.45,431.06),(11.1507,354.81,526.80,954.08),(11.0914,354.91,526.96,954.04),
    (10.8341,273.59,57.42,57.43),(10.7714,354.81,527.04,954.00),(10.7242,272.99,57.06,57.34),
    (10.7061,858.65,629.58,431.16),(10.6996,858.28,628.88,431.76),(10.5956,353.49,537.33,956.16),
    (10.5779,470.66,828.48,17.67),(10.5683,858.17,629.34,431.51),(10.5461,858.55,629.54,431.22),
    (10.5353,353.32,537.11,955.83),(10.5067,470.86,828.30,17.46),(10.4794,471.58,828.40,17.82),
    (10.4374,470.92,828.08,17.04),(10.3517,470.98,827.92,16.89)]),
 "$\\Delta\\Omega_m$ 0.10  (0.26$\\to$0.36)": np.array([
    (19.0338,645.35,37.89,509.22),(19.0132,645.59,37.90,509.66),(18.9049,433.58,408.14,203.15),
    (18.1830,645.55,37.58,509.48),(18.1606,404.94,3.39,553.82),(17.9342,897.78,928.31,592.50),
    (17.8682,645.39,38.02,509.53),(17.8015,645.56,37.59,509.54),(17.7846,628.30,441.73,116.46),
    (17.7608,645.32,38.05,509.49),(17.7476,808.12,356.31,556.64),(17.5204,404.64,180.28,850.26),
    (17.4467,645.53,37.74,509.44),(17.3502,897.79,928.34,592.27),(17.3080,280.31,162.84,410.22),
    (17.2537,499.35,704.31,417.14),(17.2452,280.22,162.74,410.36),(17.2387,898.02,928.39,592.58),
    (17.2143,324.39,499.37,357.46),(17.2017,279.65,162.44,411.20)]),
 "$\\Delta\\Omega_m$ 0.15  (0.21$\\to$0.36)": np.array([
    (19.8687,432.62,407.79,203.64),(18.3503,646.80,37.99,508.50),(18.0228,897.90,928.49,592.51),
    (17.9585,646.89,38.06,508.36),(17.9165,897.77,928.38,592.59),(17.8261,499.67,703.28,417.09),
    (17.8224,499.70,703.33,416.77),(17.7712,499.71,703.38,416.74),(17.6832,646.40,37.94,508.63),
    (17.6360,499.73,703.29,416.81),(17.6207,499.61,703.33,416.76),(17.6148,898.14,927.63,593.38),
    (17.5887,499.60,703.27,416.93),(17.5403,499.67,703.32,416.79),(17.5146,499.73,703.31,416.75),
    (17.4750,499.72,703.45,416.81),(17.4079,499.70,703.36,416.71),(17.3706,499.79,703.34,416.91),
    (17.3686,646.37,37.95,508.63),(17.3671,499.66,703.30,416.74)]),
 "$\\Delta\\Omega_m$ 0.15, $w_0=-1.4$": np.array([
    (18.8452,764.77,812.91,930.15),(18.4853,68.39,277.33,64.02),(17.4942,762.95,813.67,929.16),
    (17.3506,629.74,441.08,117.50),(17.1111,762.40,813.51,928.92),(16.8583,629.20,441.96,117.79),
    (16.8258,629.34,441.04,117.84),(16.5215,408.15,181.98,852.42),(16.5168,408.20,182.04,852.39),
    (16.4971,416.87,189.02,849.04),(16.4947,408.12,182.15,852.32),(16.4921,408.13,182.03,852.37),
    (16.2655,418.08,189.50,848.59),(16.2042,761.98,813.90,928.86),(16.1810,761.98,814.10,928.76),
    (16.0632,408.18,182.02,852.39),(15.9587,141.45,697.25,818.32),(15.9316,130.22,854.66,906.89),
    (15.8961,724.39,379.64,444.65),(15.8753,801.20,20.83,777.41)]),
}

allfull = np.concatenate(list(FULL.values()))

fig = plt.figure(figsize=(15.5, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.28)

# ---------- top row: x-z, y-z for smoke vs full, plus the z histogram ----------
for col, (lab, ia) in enumerate([("x", 1), ("y", 2)]):
    ax = fig.add_subplot(gs[0, col])
    ax.scatter(allfull[:, ia], allfull[:, 3], s=26, color="#2c7fb8",
               edgecolor="black", lw=0.35, label="6 full-box scans (250/250), n=%d" % len(allfull))
    ax.scatter(SMOKE[:, ia], SMOKE[:, 3], s=52, marker="^", color="#d62728",
               edgecolor="black", lw=0.5, label="8-subfile smoke, n=20")
    ax.axhspan(0, 34, color="#d62728", alpha=0.10, zorder=0)
    ax.text(BOX * 0.98, 44, "the smoke could only see this band\n(8 of 250 subfiles = z 0-34)",
            ha="right", fontsize=8.5, color="#a01f1f")
    ax.set_xlabel("%s [cMpc/h]" % lab); ax.set_ylabel("z [cMpc/h]")
    ax.set_xlim(0, BOX); ax.set_ylim(0, BOX); ax.grid(alpha=0.22)
    ax.set_title("%s-z projection" % lab, fontsize=12)
    if col == 0:
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

ax = fig.add_subplot(gs[0, 2])
b = np.linspace(0, BOX, 26)
ax.hist(allfull[:, 3], bins=b, color="#2c7fb8", alpha=0.85, label="full-box scans")
ax.hist(SMOKE[:, 3], bins=b, color="#d62728", alpha=0.85, label="smoke")
ax.axvspan(0, 34, color="#d62728", alpha=0.12, zorder=0)
ax.set_xlabel("z [cMpc/h]"); ax.set_ylabel("movers per bin")
ax.set_title("z distribution of the top movers", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.22, axis="y")

# ---------- bottom row: x-y, smoke and full together ----------
ax = fig.add_subplot(gs[1, :2])
cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(FULL)))
for (name, d), c in zip(FULL.items(), cmap):
    ax.scatter(d[:, 1], d[:, 2], s=52, color=c, edgecolor="black", lw=0.4, label=name)
ax.scatter(SMOKE[:, 1], SMOKE[:, 2], s=62, marker="^", color="#d62728",
           edgecolor="black", lw=0.5, label="8-subfile smoke", zorder=4)
# the corner pile the smoke reported, which prompted the periodic-boundary question
cl = SMOKE[(SMOKE[:, 1] < 10) & (SMOKE[:, 2] > 975)]
ax.add_patch(plt.Circle((cl[:, 1].mean(), cl[:, 2].mean()), 55, fill=False,
                        color="#d62728", lw=1.6, ls="--", zorder=5))
ax.annotate("%d of the smoke's top 20 sit here,\nat (%.0f, %.0f) — near the box corner"
            % (len(cl), cl[:, 1].mean(), cl[:, 2].mean()),
            xy=(cl[:, 1].mean() + 45, cl[:, 2].mean()), xytext=(210, 900),
            fontsize=9, color="#a01f1f",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3))
ax.set_xlabel("x [cMpc/h]"); ax.set_ylabel("y [cMpc/h]")
ax.set_xlim(0, BOX); ax.set_ylim(0, BOX); ax.set_aspect("equal"); ax.grid(alpha=0.22)
ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95, ncol=2)
ax.set_title("x-y: the smoke's corner pile vs the full scans spread over the box",
             fontsize=12)

# ---------- bottom right: how tightly they cluster ----------
ax = fig.add_subplot(gs[1, 2])
cell = 20.0
names, frac = [], []
for name, d in FULL.items():
    key = np.floor(d[:, 1:] / cell).astype(int)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    names.append(name); frac.append(cnt.max() / len(d))
y = np.arange(len(names))
ax.barh(y, frac, color=cmap, edgecolor="black", lw=0.5)
for i, f in enumerate(frac):
    ax.text(f + 0.01, i, "%d/20" % round(f * 20), va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([n.replace("$", "").replace("\\Delta\\Omega_m", "dOm")
                    .replace("\\Omega_m", "Om").replace("\\to", "->")
                    .replace("$w_0w_a$", "w0wa") for n in names], fontsize=8)
ax.set_xlabel("largest share of the top-20\nin a single 20 cMpc/h cell")
ax.set_xlim(0, 1.0); ax.grid(alpha=0.22, axis="x")
ax.set_title("movers cluster on halos", fontsize=11.5)

fig.suptitle("The movers were never piled up at 0 — that was the smoke test's truncated z-range",
             fontsize=13.5, y=0.965)
out = "/Users/hyunohyeo/Desktop/movers_compare.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("smoke  z range: %6.2f .. %7.2f  (n=%d)" % (SMOKE[:, 3].min(), SMOKE[:, 3].max(), len(SMOKE)))
print("full   z range: %6.2f .. %7.2f  (n=%d, %d scans)"
      % (allfull[:, 3].min(), allfull[:, 3].max(), len(allfull), len(FULL)))
print("full   x range: %6.2f .. %7.2f ; y range: %6.2f .. %7.2f"
      % (allfull[:, 1].min(), allfull[:, 1].max(), allfull[:, 2].min(), allfull[:, 2].max()))
