"""Analyze 03_nongaussian_d3.csv: per (shape, snr), compare D2 vs D3
across selection rules -- including 1-SE-style rules (largest lam whose
AIC criterion is within k*sqrt(2N) of its minimum) -- on max core
residual AND h3/h4 moment bias (fit minus truth-curve-through-same-
pipeline). The question: does D3 + a stiff rule erase real h3/h4?
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "03_nongaussian_d3.csv"))
N_PIX = 444
SE = np.sqrt(2 * N_PIX)


def pick(g, rule):
    lams = np.array(sorted(g.index))
    chi2 = g.loc[lams, "chi2"].to_numpy()
    m = g.loc[lams, "m_eff"].to_numpy()
    crit = chi2 + 2.0 * m
    if rule.startswith("aic_w"):
        w_m = float(rule[5:])
        return lams[np.argmin(chi2 + w_m * m)]
    if rule.startswith("aic1se_k"):
        k = float(rule[8:])
        return lams[crit <= crit.min() + k * SE].max()
    if rule == "disc_sqrt2n":
        return lams[chi2 <= chi2.min() + SE].max()
    if rule == "oracle":
        return lams[np.argmin(g.loc[lams, "max_resid_pct"].to_numpy())]
    raise ValueError(rule)


RULES = ["aic_w2", "aic_w8", "aic1se_k0.25", "aic1se_k0.5", "aic1se_k1.0",
         "disc_sqrt2n", "oracle"]

pd.set_option("display.width", 240)
out_rows = []
for (h3t, h4t, snr, order), gg in df.groupby(["h3_true", "h4_true", "snr", "order"]):
    for rule in RULES:
        res, h3b, h4b, sigb, lams = [], [], [], [], []
        for seed, g in gg.groupby("seed"):
            g = g.set_index("lam").sort_index()
            lam = pick(g, rule)
            res.append(g.loc[lam, "max_resid_pct"])
            h3b.append(g.loc[lam, "h3_fit"] - g.loc[lam, "h3_truthcurve"])
            h4b.append(g.loc[lam, "h4_fit"] - g.loc[lam, "h4_truthcurve"])
            sigb.append(g.loc[lam, "sigma_mom"] - 50.0)
            lams.append(lam)
        res, h3b, h4b, sigb = map(np.array, (res, h3b, h4b, sigb))
        out_rows.append(dict(
            h3_true=h3t, h4_true=h4t, snr=snr, order=order, rule=rule,
            mean_res=res.mean(), med_res=np.median(res), max_res=res.max(),
            n_lt5=(res < 5).sum(), n=len(res),
            mean_h3_bias=h3b.mean(), mean_h4_bias=h4b.mean(),
            mean_sig_bias=sigb.mean(), med_lam=np.median(lams),
        ))
summary = pd.DataFrame(out_rows)
summary.to_csv(os.path.join(HERE, "03_rule_summary.csv"), index=False)

for (h3t, h4t), block in summary.groupby(["h3_true", "h4_true"]):
    print(f"\n===== truth h3={h3t} h4={h4t} =====")
    for snr, sb in block.groupby("snr"):
        print(f"\n  S/N={snr:.0f}")
        print("  order rule           mean_res med_res max_res n<5  h3_bias  h4_bias  sig_bias med_lam")
        for _, r in sb.sort_values(["order", "rule"]).iterrows():
            print(f"  D{r['order']:.0f}   {r['rule']:14s} {r['mean_res']:7.2f} {r['med_res']:7.2f} "
                  f"{r['max_res']:7.2f} {r['n_lt5']:3.0f} {r['mean_h3_bias']:+8.4f} {r['mean_h4_bias']:+8.4f} "
                  f"{r['mean_sig_bias']:+8.2f} {r['med_lam']:8.4g}")
