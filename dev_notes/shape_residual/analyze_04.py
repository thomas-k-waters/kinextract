"""Analyze 04_sigma_range_d3.csv: for each (sigma, snr, scale_pow),
where does the oracle lam land, and how do residual/sigma-bias at the
oracle and at the AIC pick behave across sigma? The scaling exponent
whose oracle-lam distribution stays put across sigma wins.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "04_sigma_range_d3.csv"))
pd.set_option("display.width", 240)

rows = []
for (sig, snr, sp), gg in df.groupby(["sigma_true", "snr", "scale_pow"]):
    lam_oracles, res_o, res_aic, lam_aics, sig_bias_aic = [], [], [], [], []
    for seed, g in gg.groupby("seed"):
        g = g.set_index("lam").sort_index()
        lo = g["max_resid_pct"].idxmin()
        la = (g["chi2"] + 2 * g["m_eff"]).idxmin()
        lam_oracles.append(lo)
        lam_aics.append(la)
        res_o.append(g.loc[lo, "max_resid_pct"])
        res_aic.append(g.loc[la, "max_resid_pct"])
        sig_bias_aic.append(g.loc[la, "sigma_mom"] - sig)
    rows.append(dict(
        sigma=sig, snr=snr, scale_pow=sp,
        med_lam_oracle=np.median(lam_oracles), med_lam_aic=np.median(lam_aics),
        mean_res_oracle=np.mean(res_o), mean_res_aic=np.mean(res_aic),
        max_res_aic=np.max(res_aic), n_lt5_aic=(np.array(res_aic) < 5).sum(),
        n_lt5_oracle=(np.array(res_o) < 5).sum(),
        mean_sig_bias_aic=np.mean(sig_bias_aic),
        n_knots=gg["n_knots"].iloc[0], dv=gg["dv_knot"].iloc[0],
    ))
out = pd.DataFrame(rows).sort_values(["scale_pow", "snr", "sigma"])
for sp, g in out.groupby("scale_pow"):
    print(f"\n=== scale_pow={sp} ===")
    print(g.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
out.to_csv(os.path.join(HERE, "04_summary.csv"), index=False)
