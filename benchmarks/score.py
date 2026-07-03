"""Bias/scatter scoring and threshold flags for the kinextract validation suite.

THRESHOLDS below define what "the code falls short" means operationally.
They are explicit proposals, not first-principles requirements -- adjust
them to match your own science precision requirements before treating the
"flagged scenarios" report table as authoritative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = {
    # |mean bias in V| exceeding this fraction of sigma_true is flagged.
    "bias_v_frac_of_sigma": 0.05,
    # |mean bias in sigma| exceeding this fraction of sigma_true is flagged.
    # Calibrated to the magnitude of the already-observed real NGC 5102
    # Fortran-vs-kinextract disagreement (~10% fractional).
    "bias_sigma_frac_of_sigma": 0.10,
    # |mean bias in h3| or |h4| exceeding this absolute value is flagged
    # (h3/h4 truth values on the LOSVD-shape axis are O(0.05-0.15)).
    "bias_h3h4_abs": 0.03,
    # mean_bias / (std_bias / sqrt(n_replicates)) -- bias significantly
    # nonzero relative to noise-realization scatter (systematic, not a
    # single unlucky draw).
    "bias_significance_nsigma": 2.0,
    # chi2_red outside this range flags a poor continuum/model match
    # (over- or under-fitting), independent of whether moments look biased.
    "chi2_red_low": 0.85,
    "chi2_red_high": 1.20,
}


def aggregate_replicates(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Group by (scenario_id, continuum_mode); compute mean/std bias and
    chi2_red, crash/failure counts, and mean wall-time. One row per group.
    """
    df = raw_df[~raw_df.get("_header", False).astype(bool)] if "_header" in raw_df else raw_df
    ok = df[df["crashed"] == False].copy()  # noqa: E712

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        out = {"n_replicates": n}
        for col in ("v", "sigma", "h3", "h4"):
            bias = g[f"bias_{col}"].astype(float)
            out[f"mean_bias_{col}"] = bias.mean()
            out[f"std_bias_{col}"] = bias.std(ddof=1) if n > 1 else 0.0
        out["mean_chi2_red"] = g["chi2_red"].astype(float).mean()
        out["mean_wall_time_s"] = g["wall_time_s"].astype(float).mean()
        out["max_wall_time_s"] = g["wall_time_s"].astype(float).max()
        return pd.Series(out)

    agg = ok.groupby(["scenario_id", "axis", "continuum_mode"], as_index=False).apply(
        _agg, include_groups=False
    )
    crash_counts = df.groupby(["scenario_id", "continuum_mode"], as_index=False)["crashed"].sum()
    crash_counts = crash_counts.rename(columns={"crashed": "n_crashed"})
    agg = agg.merge(crash_counts, on=["scenario_id", "continuum_mode"], how="left")

    truth_cols = df.drop_duplicates("scenario_id")[["scenario_id", "sigma_true", "v_true"]]
    agg = agg.merge(truth_cols, on="scenario_id", how="left")
    return agg


def flag_scenario(row: pd.Series) -> dict:
    """Apply THRESHOLDS to one aggregated row; return flags + reason strings."""
    sigma_true = max(float(row["sigma_true"]), 1e-6)
    n = max(int(row["n_replicates"]), 1)
    reasons = []

    for moment, thresh_key, scale in (
        ("v", "bias_v_frac_of_sigma", sigma_true),
        ("sigma", "bias_sigma_frac_of_sigma", sigma_true),
    ):
        mean_b = row[f"mean_bias_{moment}"]
        std_b = row[f"std_bias_{moment}"]
        sem = std_b / np.sqrt(n) if n > 1 else np.inf
        significant = abs(mean_b) / sem > THRESHOLDS["bias_significance_nsigma"] if sem > 0 else False
        exceeds = abs(mean_b) / scale > THRESHOLDS[thresh_key]
        if exceeds and significant:
            reasons.append(
                f"bias_{moment}={mean_b:+.2f} ({abs(mean_b)/scale:.1%} of sigma_true), "
                f"{abs(mean_b)/sem:.1f}-sigma significant"
            )

    for moment in ("h3", "h4"):
        mean_b = row[f"mean_bias_{moment}"]
        std_b = row[f"std_bias_{moment}"]
        sem = std_b / np.sqrt(n) if n > 1 else np.inf
        significant = abs(mean_b) / sem > THRESHOLDS["bias_significance_nsigma"] if sem > 0 else False
        if abs(mean_b) > THRESHOLDS["bias_h3h4_abs"] and significant:
            reasons.append(f"bias_{moment}={mean_b:+.4f}, {abs(mean_b)/sem:.1f}-sigma significant")

    chi2r = row["mean_chi2_red"]
    if chi2r < THRESHOLDS["chi2_red_low"] or chi2r > THRESHOLDS["chi2_red_high"]:
        reasons.append(f"mean chi2_red={chi2r:.2f} outside [{THRESHOLDS['chi2_red_low']}, "
                        f"{THRESHOLDS['chi2_red_high']}]")

    if row["n_crashed"] > 0:
        reasons.append(f"{int(row['n_crashed'])} replicate(s) crashed")

    return {"flagged": len(reasons) > 0, "reasons": "; ".join(reasons)}
