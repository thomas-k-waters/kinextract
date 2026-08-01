"""Recovered-vs-true curves for the D3 penalty (from 02's NPZ) at
representative cases, next to the D2 default at lam=2 (from 01's NPZ).
Panels: S/N=30 seed 2 (bad D2 seed per the handoff), S/N=60 seed 0
(the canonical case), S/N=100 seed 0.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d2 = np.load(os.path.join(HERE, "01_oracle_lambda_curves.npz"))
d3 = np.load(os.path.join(HERE, "02_penalty_order_d3_curves.npz"))
v = d2["v_grid"]

CASES = [(30, 2), (60, 0), (100, 0)]
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08, "wspace": 0.22})
for j, (snr, seed) in enumerate(CASES):
    truth = d2[f"truth_{snr}_{seed}"]
    peak = truth.max()
    core = truth > 0.01 * peak
    series = [(f"D2 lam=2 (current default form)", d2[f"b_{snr}_{seed}_2.0"], "#D55E00"),
              (f"D3 lam=1024", d3[f"b_{snr}_{seed}_1024.0"], "#0072B2"),
              (f"D3 lam=16384 (Gaussian limit)", d3[f"b_{snr}_{seed}_16384.0"], "#009E73")]
    ax = axes[0, j]
    ax.plot(v, truth, color="#1a1a1a", lw=2.2, label="truth", zorder=5)
    for label, b, c in series:
        r = np.abs(b - truth)[core].max() / peak * 100
        ax.plot(v, b, color=c, lw=1.5, label=f"{label} [{r:.1f}%]")
    ax.set_title(f"S/N={snr}, seed={seed}", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    if j == 0:
        ax.set_ylabel("b(v)")
    ax = axes[1, j]
    for label, b, c in series:
        ax.plot(v, (b - truth) / peak * 100, color=c, lw=1.4)
    ax.axhline(0, color="#1a1a1a", lw=0.8)
    for y in (5, -5):
        ax.axhline(y, color="#999999", lw=0.8, ls="--")
    vc = v[core]
    ax.axvspan(vc.min(), vc.max(), color="#000000", alpha=0.05, lw=0)
    ax.set_xlabel("v (km/s)")
    ax.set_ylim(-14, 14)
    if j == 0:
        ax.set_ylabel("signed resid (% peak)")
for a in axes.flat:
    a.grid(alpha=0.2, lw=0.5)
    a.spines[["top", "right"]].set_visible(False)
out = os.path.join(HERE, "02_d3_curves.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(out)
