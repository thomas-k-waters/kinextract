"""Analyze 06_final_validation.csv: paired comparison of the current
default (D2+aic_local) vs candidate (D3+aic_local, extended grid),
per (sigma, snr, shape, recipe) cell.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "06_final_validation.csv"))
pd.set_option("display.width", 260)

df["sig_bias"] = df["sigma_mom"] - df["sherm_truthcurve"]
df["V_bias"] = df["V"] - df["V_truthcurve"]
df["h3_bias"] = df["h3_fit"] - df["h3_truthcurve"]
df["h4_bias"] = df["h4_fit"] - df["h4_truthcurve"]

rows = []
for (sig, snr, h3t, h4t, rec), g in df.groupby(["sigma_true", "snr", "h3_true", "h4_true", "recipe"]):
    piv = g.pivot_table(index="seed", columns="method",
                        values=["max_resid_pct", "sig_bias", "h3_bias", "h4_bias",
                                "wing_mass_ratio", "runtime_s"])
    d2 = piv.xs("default_d2_aic", axis=1, level=1)
    d3 = piv.xs("candidate_d3_aic", axis=1, level=1)
    rows.append(dict(
        sigma=sig, snr=snr, h3=h3t, h4=h4t, recipe=rec, n=len(d2),
        res_d2_mean=d2["max_resid_pct"].mean(), res_d3_mean=d3["max_resid_pct"].mean(),
        res_d2_med=d2["max_resid_pct"].median(), res_d3_med=d3["max_resid_pct"].median(),
        res_d2_max=d2["max_resid_pct"].max(), res_d3_max=d3["max_resid_pct"].max(),
        n5_d2=(d2["max_resid_pct"] < 5).sum(), n5_d3=(d3["max_resid_pct"] < 5).sum(),
        d3_wins=(d3["max_resid_pct"] < d2["max_resid_pct"]).sum(),
        sigb_d2=d2["sig_bias"].mean(), sigb_d3=d3["sig_bias"].mean(),
        h4b_d2=d2["h4_bias"].mean(), h4b_d3=d3["h4_bias"].mean(),
        h3b_d2=d2["h3_bias"].mean(), h3b_d3=d3["h3_bias"].mean(),
        wing_d2=d2["wing_mass_ratio"].median(), wing_d3=d3["wing_mass_ratio"].median(),
        t_d2=d2["runtime_s"].median(), t_d3=d3["runtime_s"].median(),
    ))
out = pd.DataFrame(rows).sort_values(["recipe", "h4", "snr", "sigma"])
fmt = lambda x: f"{x:.2f}"
for (rec, h3t, h4t), g in out.groupby(["recipe", "h3", "h4"]):
    print(f"\n===== recipe={rec}  truth h3={h3t} h4={h4t} =====")
    cols = ["sigma", "snr", "n", "res_d2_mean", "res_d3_mean", "res_d2_med", "res_d3_med",
            "res_d2_max", "res_d3_max", "n5_d2", "n5_d3", "d3_wins",
            "sigb_d2", "sigb_d3", "h4b_d2", "h4b_d3", "wing_d2", "wing_d3", "t_d2", "t_d3"]
    print(g[cols].to_string(index=False, float_format=fmt))

print("\n=== overall ===")
for m in ["res_d2_mean", "res_d3_mean"]:
    print(f"{m}: {out[m].mean():.2f}")
print(f"total paired wins for D3: {out['d3_wins'].sum()} / {out['n'].sum()}")
print(f"cells where D3 mean better: {(out['res_d3_mean'] < out['res_d2_mean']).sum()}/{len(out)}")
print(f"|sigma bias|: D2 {out['sigb_d2'].abs().mean():.2f}  D3 {out['sigb_d3'].abs().mean():.2f}")
print(f"|h4 bias|: D2 {out['h4b_d2'].abs().mean():.4f}  D3 {out['h4b_d3'].abs().mean():.4f}")
out.to_csv(os.path.join(HERE, "06_summary.csv"), index=False)
