"""Analyze 01_oracle_lambda_sweep.csv: per (snr, seed), compare
- residual at the AIC-local pick (over the default lam grid),
- residual at the oracle lam (min max-core-resid over the extended grid),
- residual at fixed lam=2.0,
and summarize the selection gap. Also checks curvature fit-vs-truth and
objective_gap correlation with residual.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "01_oracle_lambda_sweep.csv"))
DEFAULT_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

rows = []
for (snr, seed), g in df.groupby(["snr", "seed"]):
    g = g.set_index("lam")
    gd = g.loc[DEFAULT_GRID]
    lam_aic = gd["criterion"].idxmin()
    lam_oracle = g["max_resid_pct"].idxmin()
    rows.append(dict(
        snr=snr, seed=seed, lam_aic=lam_aic, lam_oracle=lam_oracle,
        res_aic=g.loc[lam_aic, "max_resid_pct"],
        res_oracle=g.loc[lam_oracle, "max_resid_pct"],
        res_lam2=g.loc[2.0, "max_resid_pct"],
        frac_aic=g.loc[lam_aic, "frac_over5_pct"],
        frac_oracle=g.loc[lam_oracle, "frac_over5_pct"],
        frac_lam2=g.loc[2.0, "frac_over5_pct"],
        sigma_aic=g.loc[lam_aic, "sigma_mom"], sigma_oracle=g.loc[lam_oracle, "sigma_mom"],
        sigma_lam2=g.loc[2.0, "sigma_mom"],
        curv_ratio_aic=g.loc[lam_aic, "curv_fit"] / g.loc[lam_aic, "curv_truth"],
        objgap_aic=g.loc[lam_aic, "objective_gap"],
    ))
per = pd.DataFrame(rows)

pd.set_option("display.width", 200)
print("=== per-(snr,seed) selection comparison ===")
print(per.to_string(index=False))

print("\n=== summary by S/N (mean / max over seeds) ===")
for snr, g in per.groupby("snr"):
    print(f"\nS/N={snr:.0f}  (n={len(g)})")
    for col in ["res_aic", "res_lam2", "res_oracle"]:
        print(f"  {col:10s}: mean {g[col].mean():5.2f}%  median {g[col].median():5.2f}%  max {g[col].max():5.2f}%   frac_cases>5%: {(g[col]>5).mean()*100:4.0f}%")
    print(f"  lam_aic   : {sorted(g['lam_aic'].tolist())}")
    print(f"  lam_oracle: {sorted(g['lam_oracle'].tolist())}")
    print(f"  curv_ratio at aic pick (fit/truth): median {g['curv_ratio_aic'].median():.2f}")

print("\n=== does oracle lam hit <5% everywhere? ===")
for snr, g in per.groupby("snr"):
    n_ok = (g["res_oracle"] < 5).sum()
    print(f"S/N={snr:.0f}: oracle <5% in {n_ok}/{len(g)} seeds; AIC <5% in {(g['res_aic']<5).sum()}/{len(g)}; lam2 <5% in {(g['res_lam2']<5).sum()}/{len(g)}")

per.to_csv(os.path.join(HERE, "01_per_case_summary.csv"), index=False)

# --- alternative selection rules, evaluated offline from (chi2, m_eff) ---
N_PIX = 444  # fit window 8400-8800 A, single E-MILES template (checked directly)

def pick(g, rule):
    """g indexed by lam (ascending). Returns chosen lam."""
    lams = np.array(sorted(g.index))
    chi2 = g.loc[lams, "chi2"].to_numpy()
    m_eff = g.loc[lams, "m_eff"].to_numpy()
    if rule.startswith("aic_w"):
        w_m = float(rule.split("aic_w")[1])
        return lams[np.argmin(chi2 + w_m * m_eff)]
    if rule == "gcv":
        return lams[np.argmin(chi2 / (N_PIX - m_eff) ** 2)]
    if rule == "discrepancy_sqrt2n":  # pPXF-style: largest lam with chi2 <= chi2_min + sqrt(2N)
        ok = chi2 <= chi2.min() + np.sqrt(2.0 * N_PIX)
        return lams[ok].max()
    if rule == "discrepancy_abs":  # Morozov: largest lam with chi2 <= N
        ok = chi2 <= N_PIX
        return lams[ok].max() if ok.any() else lams[np.argmin(chi2)]
    raise ValueError(rule)

RULES = ["aic_w2", "aic_w4", "aic_w8", "gcv", "discrepancy_sqrt2n", "discrepancy_abs"]
print("\n=== alternative selection rules (extended grid available to all) ===")
alt_rows = []
for (snr, seed), g in df.groupby(["snr", "seed"]):
    g = g.set_index("lam").sort_index()
    row = dict(snr=snr, seed=seed, res_oracle=g["max_resid_pct"].min())
    for rule in RULES:
        lam = pick(g, rule)
        row[f"lam_{rule}"] = lam
        row[f"res_{rule}"] = g.loc[lam, "max_resid_pct"]
    alt_rows.append(row)
alt = pd.DataFrame(alt_rows)
for snr, g in alt.groupby("snr"):
    print(f"\nS/N={snr:.0f}")
    for rule in RULES + ["oracle"]:
        col = f"res_{rule}" if rule != "oracle" else "res_oracle"
        lamcol = f"lam_{rule}" if rule != "oracle" else None
        lams = sorted(g[lamcol].tolist()) if lamcol else []
        print(f"  {rule:20s}: mean {g[col].mean():5.2f}%  median {g[col].median():5.2f}%  "
              f"max {g[col].max():6.2f}%  n<5%: {(g[col]<5).sum():2d}/{len(g)}  lams {lams}")
alt.to_csv(os.path.join(HERE, "01_alt_rules_summary.csv"), index=False)
