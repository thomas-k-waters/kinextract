"""Re-validation re-run of sweep_nbins.py's nbins=29 grid on current code.

This session found and fixed a regression: n_losvd_bins had been changed
29 -> 89 without recalibrating the auto-xlam discrepancy-principle search,
and a separate notebook-only xlam_discrepancy_nsigma=1.0 override
interacted with it catastrophically. n_losvd_bins is now reverted to 29
(the shipped default) and the notebook override is gone -- but several
other unrelated changes landed this session (E-MILES template-packing
support, the bracket-expansion warning, joint.py's coeff_scale JIT-caching
refactor, dead-code removal). This script re-runs the ORIGINAL
sweep_nbins.py sigma grid (30-350 km/s, 3 seeds), restricted to nbins=29
(the only bin count actually shipped now), on the current codebase, and
saves fresh results alongside (not overwriting) the original
results_nbins_sweep.json baseline -- so accuracy/timing can be diffed
against that baseline by compare_nbins_revalidate.py, confirming nothing
*else* regressed independent of the already-root-caused nbins/nsigma
interaction.
"""
import sys, time, json
sys.path.insert(0, "dev_notes/stress_test")
from harness_emiles import build_emiles_mock, write_spec_file, WAVEMIN
from kinextract import FitConfig, run_spectral_fit, set_verbose
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite
import numpy as np
import tempfile

EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"


def fit_kinextract(spec_path, sigma_guess, n_losvd_bins):
    vrange = max(300.0, 3.5 * sigma_guess)
    cfg = FitConfig(
        template_list_file=EMILES_TLIST, template_dir=EMILES_DIR,
        wavemin_full=WAVEMIN, step=1.25, wavefitmin=8000.0, wavefitmax=9000.0,
        zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
        losvd_vmin=-vrange, losvd_vmax=vrange, sigl=sigma_guess, clean=False,
        map_maxiter=10000, print_every=999999, n_losvd_bins=n_losvd_bins,
    )
    t0 = time.time()
    fit = run_spectral_fit(cfg, gal_file=str(spec_path))
    dt = time.time() - t0
    gh = fit_losvd_gauss_hermite(fit["state"].xl, fit["outputs"]["b"], fit_h3h4=True)
    return dict(v=gh["vherm"], sigma_rec=gh["sherm"], chi2_red=fit["outputs"]["chi2_red"],
                time=dt, success=bool(fit["result"].success), xlam=float(fit["state"].xlam))


if __name__ == "__main__":
    sigmas = [30, 50, 70, 90, 120, 160, 200, 250, 300, 350]
    seeds = [42, 1, 7]
    true_v = 80.0
    nbins = 29  # the only bin count actually shipped now -- see config.py's n_losvd_bins docstring
    tmpdir = tempfile.mkdtemp()
    results = []
    t0 = time.time()
    print(f"{'sigma':>6s} {'seed':>5s} {'nbins':>6s} {'V':>8s} {'sigma_rec':>9s} {'chi2_red':>9s} {'xlam':>10s} {'ok':>5s} {'time':>7s}")
    for sigma in sigmas:
        for seed in seeds:
            gal, errs = build_emiles_mock(true_v, sigma, seed)
            spec_path = f"{tmpdir}/mock_{sigma}_{seed}.spec"
            write_spec_file(gal, errs, spec_path)
            r = fit_kinextract(spec_path, sigma, nbins)
            row = dict(sigma=sigma, seed=seed, nbins=nbins, true_v=true_v, **r)
            results.append(row)
            print(f"{sigma:6.0f} {seed:5d} {nbins:6d} {r['v']:8.2f} {r['sigma_rec']:9.2f} "
                  f"{r['chi2_red']:9.3f} {r['xlam']:10.3g} {str(r['success']):>5s} {r['time']:7.1f}s")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"\n{'sigma':>6s} {'Vbias':>8s} {'Vstd':>7s} {'sigbias':>8s} {'sigstd':>7s} {'avg_time':>9s} {'n_fail':>7s}")
    for sigma in sigmas:
        rows = [r for r in results if r["sigma"] == sigma]
        vb = np.array([r["v"] for r in rows]) - true_v
        sb = np.array([r["sigma_rec"] for r in rows]) - sigma
        tt = np.array([r["time"] for r in rows])
        n_fail = sum(1 for r in rows if not r["success"])
        print(f"{sigma:6.0f} {vb.mean():+8.2f} {vb.std():7.2f} {sb.mean():+8.2f} {sb.std():7.2f} {tt.mean():9.1f} {n_fail:7d}")

    with open("dev_notes/stress_test/results_nbins_sweep_revalidate.json", "w") as f:
        json.dump(results, f, indent=2)
