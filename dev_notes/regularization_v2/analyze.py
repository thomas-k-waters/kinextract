"""Summaries of bench.py results: python analyze.py results/screen_*.csv"""
import os
import sys
import numpy as np
import pandas as pd

df = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)
df["abs_dsig"] = df["dsig_frac"].abs()
agg = dict(n=("rep", "size"), shape_rms=("shape_rms", "median"), chi2_spmc=("chi2_spmc", "median"),
           chi2_post=("chi2_post", "median"), cover_post=("cover_post", "mean"), dsig_med=("dsig_frac", "median"),
           dsig_mad=("abs_dsig", "median"), dh4_med=("dh4", "median"), dh4_rms=("dh4", lambda x: np.sqrt(np.mean(x ** 2))),
           lam_med=("lam_used", "median"))
if "chi2_tot" in df:  # template-spread runs (terr=1)
    agg.update(chi2_tot=("chi2_tot", "median"), cover_tot=("cover_tot", "mean"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("=== failures (count against the method)")
for f in sys.argv[1:]:
    ff = f.replace(".csv", ".failed.txt")
    n = sum(1 for _ in open(ff)) if os.path.exists(ff) else 0
    print(f"  {os.path.basename(f)}: {n} failed, {len(pd.read_csv(f))} scored")
for by in (["contam", "method"], ["contam", "snr", "method"], ["contam", "shape", "method"], ["contam", "sigma", "method"]):
    print("\n=== by", by)
    print(df.groupby(by).agg(**agg).round(4).to_string())
