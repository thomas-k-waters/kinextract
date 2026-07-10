#!/usr/bin/env python3
"""Aggregate benchmarks/results/checkpoint.jsonl into a CSV + plots + report.md.

Usage
-----
    python -m benchmarks.report [--results-dir benchmarks/results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.score import aggregate_replicates, flag_scenario, THRESHOLDS


def load_raw_results(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    checkpoint_path = results_dir / "checkpoint.jsonl"
    rows, header = [], {}
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_header"):
                header = row
            else:
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "raw_results.csv", index=False)
    return df, header


def build_plots(agg_df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    axes_present = [a for a in agg_df["axis"].unique() if a not in ("base", "bug")]

    for axis in axes_present:
        sub = agg_df[(agg_df["axis"] == axis) | (agg_df["scenario_id"] == "base")]
        if sub.empty:
            continue
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"Recovery bias vs. axis: {axis}")
        for ax, moment in zip(axs.flat, ("v", "sigma", "h3", "h4")):
            for mode, color in (("joint", "tab:blue"), ("poly_amp", "tab:orange")):
                m = sub[sub["continuum_mode"] == mode].sort_values("sigma_true")
                if m.empty:
                    continue
                xkey = "sigma_true" if axis == "sigma" else "scenario_id"
                x = m[xkey] if axis == "sigma" else range(len(m))
                y = m[f"mean_bias_{moment}"]
                yerr = m[f"std_bias_{moment}"] / np.sqrt(m["n_replicates"].clip(lower=1))
                ax.errorbar(x, y, yerr=yerr, marker="o", label=mode, color=color, capsize=3)
                if axis != "sigma":
                    ax.set_xticks(range(len(m)))
                    ax.set_xticklabels(m["scenario_id"], rotation=45, ha="right", fontsize=7)
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_ylabel(f"bias_{moment}")
            ax.legend(fontsize=8)
        plt.tight_layout()
        out = plots_dir / f"bias_vs_{axis}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        paths.append(out)

    # Head-to-head bar chart: |bias_V|/sigma and |bias_sigma|/sigma, joint vs poly_amp
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    scenario_order = [s for s in agg_df["scenario_id"].unique() if s != "base"] + ["base"]
    width = 0.35
    for ax, moment, label in ((axs[0], "v", "|bias_V| / sigma_true"), (axs[1], "sigma", "|bias_sigma| / sigma_true")):
        for i, mode in enumerate(("joint", "poly_amp")):
            vals = []
            for sid in scenario_order:
                row = agg_df[(agg_df["scenario_id"] == sid) & (agg_df["continuum_mode"] == mode)]
                if row.empty:
                    vals.append(np.nan)
                else:
                    vals.append(abs(row[f"mean_bias_{moment}"].iloc[0]) / max(row["sigma_true"].iloc[0], 1e-6))
            xpos = np.arange(len(scenario_order)) + (i - 0.5) * width
            ax.bar(xpos, vals, width=width, label=mode)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    axs[1].set_xticks(range(len(scenario_order)))
    axs[1].set_xticklabels(scenario_order, rotation=60, ha="right", fontsize=7)
    plt.tight_layout()
    out = plots_dir / "head_to_head_joint_vs_poly.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    paths.append(out)

    return paths


def write_markdown_report(agg_df: pd.DataFrame, flagged_df: pd.DataFrame, header: dict,
                           n_total: int, n_crashed: int, plot_paths: list[Path], out_path: Path) -> None:
    lines = ["# kinextract validation suite -- report", ""]
    lines += [
        f"- git commit: `{header.get('git_commit', 'unknown')}`",
        f"- started: {header.get('started', 'unknown')}",
        f"- total fits: {n_total}, crashed: {n_crashed}",
        "",
    ]

    lines += ["## Headline continuum-mode comparison", ""]
    lines += ["| scenario | sigma_true | |bias_V|/sigma (joint) | |bias_V|/sigma (poly) | "
              "|bias_sigma|/sigma (joint) | |bias_sigma|/sigma (poly) | chi2_red (joint) | "
              "chi2_red (poly) | winner |",
              "|---|---|---|---|---|---|---|---|---|"]
    for sid in agg_df["scenario_id"].unique():
        rows = {m: agg_df[(agg_df["scenario_id"] == sid) & (agg_df["continuum_mode"] == m)]
                for m in ("joint", "poly_amp")}
        if any(r.empty for r in rows.values()):
            continue
        sigma_true = rows["joint"]["sigma_true"].iloc[0]
        bv = {m: abs(rows[m]["mean_bias_v"].iloc[0]) / max(sigma_true, 1e-6) for m in rows}
        bs = {m: abs(rows[m]["mean_bias_sigma"].iloc[0]) / max(sigma_true, 1e-6) for m in rows}
        c2 = {m: rows[m]["mean_chi2_red"].iloc[0] for m in rows}
        score = {m: bv[m] + bs[m] for m in rows}
        winner = "tie" if abs(score["joint"] - score["poly_amp"]) < 1e-6 else min(score, key=score.get)
        lines.append(
            f"| {sid} | {sigma_true:.0f} | {bv['joint']:.1%} | {bv['poly_amp']:.1%} | "
            f"{bs['joint']:.1%} | {bs['poly_amp']:.1%} | {c2['joint']:.2f} | {c2['poly_amp']:.2f} | {winner} |"
        )
    lines.append("")

    lines += ["## Plots", ""]
    for p in plot_paths:
        lines.append(f"![{p.stem}]({p.relative_to(out_path.parent)})")
    lines.append("")

    lines += ["## Flagged scenarios (a THRESHOLDS check tripped)", ""]
    if flagged_df.empty:
        lines.append("None.")
    else:
        lines += ["| scenario | continuum_mode | reasons |", "|---|---|---|"]
        for _, r in flagged_df.iterrows():
            lines.append(f"| {r['scenario_id']} | {r['continuum_mode']} | {r['reasons']} |")
    lines.append("")

    lines += ["## Runtime", ""]
    lines += ["| scenario | continuum_mode | mean_s | max_s |", "|---|---|---|---|"]
    for _, r in agg_df.sort_values("mean_wall_time_s", ascending=False).head(15).iterrows():
        lines.append(f"| {r['scenario_id']} | {r['continuum_mode']} | "
                      f"{r['mean_wall_time_s']:.1f} | {r['max_wall_time_s']:.1f} |")
    lines.append("")

    lines += ["## Thresholds used", "", "```", json.dumps(THRESHOLDS, indent=2), "```", ""]

    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    raw_df, header = load_raw_results(args.results_dir)
    agg_df = aggregate_replicates(raw_df)
    flags = agg_df.apply(flag_scenario, axis=1, result_type="expand")
    agg_df = pd.concat([agg_df, flags], axis=1)
    flagged_df = agg_df[agg_df["flagged"]]

    plots_dir = args.results_dir / "plots"
    plot_paths = build_plots(agg_df, plots_dir)

    report_path = args.results_dir / "report.md"
    n_total = len(raw_df[~raw_df.get("_header", False).astype(bool)]) if "_header" in raw_df else len(raw_df)
    n_crashed = int(raw_df["crashed"].sum()) if "crashed" in raw_df else 0
    write_markdown_report(agg_df, flagged_df, header, n_total, n_crashed, plot_paths, report_path)

    print(f"Report written to {report_path}")
    print(f"{len(flagged_df)} / {len(agg_df)} (scenario, continuum_mode) groups flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
