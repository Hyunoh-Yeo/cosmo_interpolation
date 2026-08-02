"""What actually drives particle displacement between cosmologies.

Four complete-box scans at z=0 (8.59 G particles each) plotted against the linear
growth difference between the two cosmologies. Dark-energy-only changes fall on a
single proportionality line; changing Omega_m does not, because it reshapes the
initial power spectrum instead of only rescaling its amplitude.

Run with a matplotlib python:  /opt/anaconda3/bin/python3 make_scaling_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "figure.dpi": 150, "axes.unicode_minus": False})

# label, dD/D [%], median |dr|, max |dr|, kind
RUNS = [
    (r"$w_0,w_a$: $(-1.2,-0.8)$ vs $(-1.0,-1.6)$", 0.28, 0.0302, 5.52, "de"),
    (r"$w_0$ only: $-1.0$ vs $-1.4$", 10.39, 1.0471, 9.96, "de"),
    (r"$\Omega_m$: 0.21 vs 0.36  ($w_0=-1$)", 14.05, 2.5119, 19.87, "om"),
    (r"$\Omega_m$: 0.21 vs 0.36  ($w_0=-1.4$)", 9.87, 2.7542, 18.85, "om"),
]
STYLE = {"de": dict(color="#2c7fb8", marker="o", label="dark energy only"),
         "om": dict(color="#d95f0e", marker="s", label=r"$\Omega_m$ changed")}
SPACING = 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ---- (a) displacement vs growth difference ----
seen = set()
for lab, dd, med, mx, kind in RUNS:
    st = dict(STYLE[kind])
    st["label"] = st["label"] if kind not in seen else None
    seen.add(kind)
    ax1.plot(dd, med, ms=12, mec="black", mew=0.7, ls="none", **st)

de = [(d, m) for _, d, m, _, k in RUNS if k == "de"]
slope = np.mean([m / d for d, m in de])
xs = np.linspace(0, 16, 50)
ax1.plot(xs, slope * xs, "--", color="#2c7fb8", lw=1.5, alpha=0.8,
         label="dark-energy scaling\n(%.3f cMpc/h per 1%%)" % slope)
for lab, dd, med, mx, kind in RUNS:
    if kind == "om":
        ax1.annotate("", xy=(dd, med), xytext=(dd, slope * dd),
                     arrowprops=dict(arrowstyle="<-", color="#d95f0e", lw=1.4))
        ax1.text(dd + 0.3, 0.5 * (med + slope * dd), "%.1f$\\times$" % (med / (slope * dd)),
                 color="#d95f0e", fontsize=10, va="center")

ax1.axhline(SPACING, color="0.6", ls=":", lw=1.3)
ax1.text(0.3, SPACING * 1.12, "interparticle spacing 0.5", fontsize=9, color="0.4")
ax1.set_xlabel(r"linear growth difference  $\Delta D/D$  from $z{=}47$ to $z{=}0$   [%]")
ax1.set_ylabel(r"median $|\Delta r|$   [cMpc/h]")
ax1.set_xlim(-0.6, 16); ax1.set_ylim(0, 3.1)
ax1.grid(alpha=0.25)
ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
ax1.set_title("(a) dark energy scales with growth; $\\Omega_m$ moves particles further",
              fontsize=12)

# ---- (b) the four runs as bars, median and max ----
labs = ["$w_0,w_a$\n(degenerate)", "$w_0$ only\n$\\Delta w_0=0.4$",
        "$\\Omega_m$\n$w_0=-1$", "$\\Omega_m$\n$w_0=-1.4$"]
med = [r[2] for r in RUNS]; mx = [r[3] for r in RUNS]
cols = [STYLE[r[4]]["color"] for r in RUNS]
x = np.arange(4)
ax2.bar(x - 0.2, med, 0.4, color=cols, edgecolor="black", lw=0.6, label="median")
ax2.bar(x + 0.2, mx, 0.4, color=cols, edgecolor="black", lw=0.6, alpha=0.45,
        hatch="//", label="max")
for i, (m, M) in enumerate(zip(med, mx)):
    ax2.text(i - 0.2, m, "%.3g" % m, ha="center", va="bottom", fontsize=9)
    ax2.text(i + 0.2, M, "%.3g" % M, ha="center", va="bottom", fontsize=9)
ax2.axhline(10, color="crimson", ls="--", lw=1.5)
ax2.text(3.45, 10.6, "10 cMpc/h", color="crimson", fontsize=9.5, ha="right")
ax2.axhline(SPACING, color="0.6", ls=":", lw=1.3)
ax2.set_yscale("log")
ax2.set_xticks(x); ax2.set_xticklabels(labs, fontsize=9.5)
ax2.set_ylabel(r"$|\Delta r|$   [cMpc/h]")
ax2.set_ylim(0.02, 40)
ax2.grid(alpha=0.25, axis="y", which="both")
ax2.legend(fontsize=9.5, loc="upper left")
ax2.set_title("(b) four full-box scans, 8.59 G particles each", fontsize=12)

fig.suptitle("Same Lagrangian particle in two cosmologies at $z=0$ — what moves it",
             fontsize=13.5, y=1.0)
out = "/Users/hyunohyeo/Desktop/dr_scaling.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("dark-energy slope: %.4f cMpc/h per 1%% growth  (from %d runs)" % (slope, len(de)))
for lab, dd, m, _, k in RUNS:
    print("  %-46s dD/D %5.2f%%  median %.4f  ratio %.3f"
          % (lab.replace("$", ""), dd, m, m / dd))
