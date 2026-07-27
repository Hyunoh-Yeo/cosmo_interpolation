"""Make two slide figures for the interpolation-methods section:
  1) methods_concept.png  — how linear / quadratic / RBF / GP each fit the same data
  2) methods_results.png  — mock accuracy comparison (off-grid + leave-one-out)
Run with a matplotlib python (e.g. /opt/anaconda3/bin/python3).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "figure.dpi": 140,
                     "font.family": "AppleGothic", "axes.unicode_minus": False})
C = {"linear": "#7f7f7f", "quadratic": "#2ca02c", "rbf": "#1f77b4", "GP": "#d62728"}

# ---------- 1) concept figure: same 5 points, 4 methods ----------
xt = np.array([-1.3, -1.0, -0.7, -0.4, -0.1])          # a cosmology parameter (e.g. w0)
g = lambda x: 8.0 + 0.7 * np.sin(2.0 * (x + 1.4))       # smooth "truth"
yt = g(xt)
xq = np.linspace(-1.3, -0.1, 300)

def rbf_pred(xq, l=0.45):
    d = xt[:, None] - xt[None, :]
    K = np.exp(-(d**2) / (2 * l**2)) + 1e-8 * np.eye(len(xt))
    c = np.linalg.solve(K, yt)
    kq = np.exp(-((xq[:, None] - xt[None, :])**2) / (2 * l**2))
    return kq @ c

def gp_pred(xq, l=0.45, sn=0.05):
    d = xt[:, None] - xt[None, :]
    K = np.exp(-(d**2) / (2 * l**2)) + sn**2 * np.eye(len(xt))
    Ki = np.linalg.inv(K)
    kq = np.exp(-((xq[:, None] - xt[None, :])**2) / (2 * l**2))
    mean = kq @ Ki @ yt
    var = 1.0 - np.einsum("ij,jk,ik->i", kq, Ki, kq)
    return mean, np.sqrt(np.clip(var, 0, None)) * 0.7

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
titles = ["linear\n(이웃 직선)", "quadratic\n(2차 곡면 fit)",
          "RBF\n(커널, 노드 통과)", "GP\n(RBF + 불확실성)"]
preds = [np.interp(xq, xt, yt),
         np.poly1d(np.polyfit(xt, yt, 2))(xq),
         rbf_pred(xq), None]
keys = ["linear", "quadratic", "rbf", "GP"]
for ax, title, key, pred in zip(axes, titles, keys, preds):
    ax.plot(xq, g(xq), "--", color="#bbbbbb", lw=1.3, label="true", zorder=1)
    if key == "GP":
        m, s = gp_pred(xq)
        ax.fill_between(xq, m - s, m + s, color=C[key], alpha=0.18, zorder=1)
        ax.plot(xq, m, color=C[key], lw=2.2, zorder=3)
    else:
        ax.plot(xq, pred, color=C[key], lw=2.2, zorder=3)
    ax.plot(xt, yt, "o", color="black", ms=6, zorder=4, label="시뮬 (node)")
    ax.set_title(title)
    ax.set_xlabel("cosmology θ (예: w0)")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("입자 변위")
axes[0].legend(loc="lower center", fontsize=9, framealpha=0.9)
fig.suptitle("네 가지 보간 방법 — 같은 데이터를 어떻게 잇는가", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("/Users/hyunohyeo/Desktop/methods_concept.png", bbox_inches="tight")
print("saved methods_concept.png")

# ---------- 2) results bar chart (mock accuracy) ----------
methods = ["linear", "quadratic", "rbf", "GP"]
offgrid = [0.590, 0.228, 0.072, 0.270]    # median pos err cMpc/h (n=48, scipy)
loo = [0.031, 0.080, 0.0008, 0.014]

fig, ax = plt.subplots(figsize=(8, 4.4))
x = np.arange(len(methods))
w = 0.38
b1 = ax.bar(x - w / 2, offgrid, w, label="격자 밖 (off-grid)",
            color=[C[m] for m in methods], edgecolor="black", lw=0.6)
b2 = ax.bar(x + w / 2, loo, w, label="leave-one-out",
            color=[C[m] for m in methods], edgecolor="black", lw=0.6, alpha=0.55, hatch="//")
ax.set_yscale("log")
ax.set_ylabel("입자별 위치오차 중앙값  [cMpc/h]  (낮을수록 좋음)")
ax.set_xticks(x); ax.set_xticklabels(["linear\n(baseline)", "quadratic", "rbf\n(tuned)", "GP"])
ax.set_title("mock 데이터 정확도 비교 (4 모델)")
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3g}", (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8)
ax.legend()
ax.grid(axis="y", alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("/Users/hyunohyeo/Desktop/methods_results.png", bbox_inches="tight")
print("saved methods_results.png")
