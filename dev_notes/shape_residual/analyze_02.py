"""Compare the D3-penalty sweep (02) against the D2 sweep (01), paired
per (snr, seed) -- same noise draws by construction. Reports oracle-lam
residuals for both penalty orders, plus what AIC-local (chi2 + 2 m_eff)
would pick under D3, and moment bias at those picks.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
d2 = pd.read_csv(os.path.join(HERE, "01_oracle_lambda_sweep.csv"))
d3 = pd.read_csv(os.path.join(HERE, "02_penalty_order_d3.csv"))

rows = []
for (snr, seed), g3 in d3.groupby(["snr", "seed"]):
    g3 = g3.set_index("lam").sort_index()
    g2 = d2[(d2.snr == snr) & (d2.seed == seed)].set_index("lam").sort_index()
    lam3_oracle = g3["max_resid_pct"].idxmin()
    lam3_aic = g3["criterion"].idxmin()
    rows.append(dict(
        snr=snr, seed=seed,
        res_d2_oracle=g2["max_resid_pct"].min(),
        res_d3_oracle=g3["max_resid_pct"].min(),
        lam3_oracle=lam3_oracle, lam3_aic=lam3_aic,
        res_d3_aic=g3.loc[lam3_aic, "max_resid_pct"],
        frac_d3_oracle=g3.loc[lam3_oracle, "frac_over5_pct"],
        sigma_d3_oracle=g3.loc[lam3_oracle, "sigma_mom"],
        sigma_d3_aic=g3.loc[lam3_aic, "sigma_mom"],
        wing_d3_oracle=g3.loc[lam3_oracle, "wing_mass_ratio"],
        wing_d3_hi=g3.loc[g3.index.max(), "wing_mass_ratio"],  # heaviest smoothing
        res_d3_hi=g3.loc[g3.index.max(), "max_resid_pct"],
    ))
per = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(per.to_string(index=False))

print("\n=== paired summary by S/N ===")
for snr, g in per.groupby("snr"):
    print(f"\nS/N={snr:.0f} (n={len(g)})")
    print(f"  D2 oracle : mean {g['res_d2_oracle'].mean():5.2f}%  median {g['res_d2_oracle'].median():5.2f}%  max {g['res_d2_oracle'].max():5.2f}%  n<5%: {(g['res_d2_oracle']<5).sum()}/{len(g)}")
    print(f"  D3 oracle : mean {g['res_d3_oracle'].mean():5.2f}%  median {g['res_d3_oracle'].median():5.2f}%  max {g['res_d3_oracle'].max():5.2f}%  n<5%: {(g['res_d3_oracle']<5).sum()}/{len(g)}")
    print(f"  D3 @AIC   : mean {g['res_d3_aic'].mean():5.2f}%  median {g['res_d3_aic'].median():5.2f}%  max {g['res_d3_aic'].max():5.2f}%  n<5%: {(g['res_d3_aic']<5).sum()}/{len(g)}")
    print(f"  D3 paired win vs D2 oracle: {(g['res_d3_oracle'] < g['res_d2_oracle']).sum()}/{len(g)} seeds")
    print(f"  lam3_oracle: {sorted(g['lam3_oracle'].tolist())}")
    print(f"  lam3_aic  : {sorted(g['lam3_aic'].tolist())}")
    print(f"  sigma at D3 oracle: mean {g['sigma_d3_oracle'].mean():6.2f} (truth 50); at D3 AIC: {g['sigma_d3_aic'].mean():6.2f}")
    print(f"  wing ratio at D3 oracle: median {g['wing_d3_oracle'].median():.2f}; at heaviest lam: {g['wing_d3_hi'].median():.2f} (resid there: median {g['res_d3_hi'].median():.2f}%)")
per.to_csv(os.path.join(HERE, "02_paired_summary.csv"), index=False)
