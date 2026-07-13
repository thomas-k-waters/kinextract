"""Full bootstrap-CI coverage test: does the reported error bar actually
contain truth, not just is the point estimate close? This is the real bar
the user set."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness_emiles import build_emiles_mock, write_spec_file  # noqa: E402

from kinextract import FitConfig, LOSVDErrorEstimator, run_spectral_fit, set_verbose  # noqa: E402
set_verbose(False)

EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"

sigmas = [40, 70, 100, 140]
seeds = [42, 1, 7]
true_v = 80.0

results = []
t0 = time.time()
for sigma in sigmas:
    vrange = max(300.0, 3.5 * sigma)
    for seed in seeds:
        gal, errs = build_emiles_mock(true_v, sigma, seed)
        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix="kinextract_cov_"))
        spec_path = tmpdir / f"mock_s{sigma}_{seed}.spec"
        write_spec_file(gal, errs, spec_path)

        cfg = FitConfig(
            template_list_file=EMILES_TLIST, template_dir=EMILES_DIR,
            wavemin_full=4750.0, step=1.25, wavefitmin=8415.0, wavefitmax=8750.0,
            zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
            losvd_vmin=-vrange, losvd_vmax=vrange, sigl=sigma, clean=False,
            map_maxiter=10000, print_every=999999,
        )
        fit = run_spectral_fit(cfg, gal_file=str(spec_path))
        est = LOSVDErrorEstimator(fit, cfg)
        boot = est.residual_bootstrap(n_bootstrap=50, n_jobs=6, seed=1)
        summary = est.summarize(bootstrap_result=boot)
        gh_map = summary["gh_map"]
        gh_lo = summary["gh_lo_recommended"]
        gh_hi = summary["gh_hi_recommended"]

        v_covered = gh_lo["gh_vherm"] <= true_v <= gh_hi["gh_vherm"]
        s_covered = gh_lo["gh_sherm"] <= sigma <= gh_hi["gh_sherm"]

        row = dict(
            sigma=sigma, seed=seed, true_v=true_v,
            gh_v=gh_map["vherm"], gh_v_lo=gh_lo["gh_vherm"], gh_v_hi=gh_hi["gh_vherm"],
            gh_sigma=gh_map["sherm"], gh_s_lo=gh_lo["gh_sherm"], gh_s_hi=gh_hi["gh_sherm"],
            v_covered=bool(v_covered), s_covered=bool(s_covered),
            chi2_red=fit["outputs"]["chi2_red"],
        )
        results.append(row)
        print(f"sigma={sigma:5.0f} seed={seed:4d}  "
              f"V={gh_map['vherm']:+7.2f} [{gh_lo['gh_vherm']:+7.2f},{gh_hi['gh_vherm']:+7.2f}] "
              f"covered={v_covered}  "
              f"sigma={gh_map['sherm']:7.2f} [{gh_lo['gh_sherm']:7.2f},{gh_hi['gh_sherm']:7.2f}] "
              f"covered={s_covered}")

n_v_covered = sum(r["v_covered"] for r in results)
n_s_covered = sum(r["s_covered"] for r in results)
print(f"\nV coverage: {n_v_covered}/{len(results)}   sigma coverage: {n_s_covered}/{len(results)}")
print(f"Total time: {time.time()-t0:.0f}s")

with open("dev_notes/stress_test/results_coverage.json", "w") as f:
    json.dump(results, f, indent=2)
