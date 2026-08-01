"""Plot the reproduction case's recovered-vs-true LOSVD curves
(00_curves.npz) -- the project rule: look at the curve before trusting
any aggregate number. Okabe-Ito CVD-safe hues, truth in near-black.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "00_curves.npz"))
v, truth, core = d["v_grid"], d["b_truth_n"], d["core"].astype(bool)

CASES = [("adaptive lam (default) -> lam=0.5", d["b_base_n"], "#0072B2"),
         ("fixed lam=2.0", d["b_fixed_n"], "#D55E00"),
         ("noiseless (adaptive -> lam=8)", d["b_nl_n"], "#009E73")]

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})
ax = axes[0]
ax.plot(v, truth, color="#1a1a1a", lw=2.2, label="truth", zorder=5)
for label, b, c in CASES:
    ax.plot(v, b, color=c, lw=1.6, label=label)
ax.set_ylabel("b(v) (unit integral)")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Single E-MILES template, V=20, sigma=50, S/N=60, seed=0", fontsize=11)

ax = axes[1]
peak = truth.max()
for label, b, c in CASES:
    ax.plot(v, (b - truth) / peak * 100, color=c, lw=1.6)
ax.axhline(0, color="#1a1a1a", lw=0.8)
for y in (5, -5):
    ax.axhline(y, color="#999999", lw=0.8, ls="--")
vc = v[core]
ax.axvspan(vc.min(), vc.max(), color="#000000", alpha=0.05, lw=0)
ax.set_ylabel("signed resid (% of peak)")
ax.set_xlabel("v (km/s)")
for a in axes:
    a.grid(alpha=0.2, lw=0.5)
    a.spines[["top", "right"]].set_visible(False)
out = os.path.join(HERE, "00_curves.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(out)
