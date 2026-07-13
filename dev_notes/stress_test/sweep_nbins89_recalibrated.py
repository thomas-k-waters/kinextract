"""Re-validation sweep for the recalibrated n_losvd_bins=89 default.

This session re-raised FitConfig.n_losvd_bins from 29 back to 89 (fixing a
real sigma-underestimate / LOSVD-peak-too-tall bias confirmed on the basic
mock, notebook 01's exact seed=42 case) after widening
FitConfig.xlam_auto_grid's upper bound from 1e5 to 1e7 -- the missing piece
last time: the auto-xlam discrepancy search needed the wider bracket to find
89 bins' much larger natural regularization strength without falling into
repeated bracket-expansion (slow, and the mechanism that ran away
catastrophically when combined with the separate notebook 02
xlam_discrepancy_nsigma=1.0 mistake, since fixed).

This script re-runs the same sigma=30-350 grid (3 seeds) as
sweep_nbins.py/sweep_nbins_revalidate.py, now at nbins=89 with the new wider
grid as an explicit default (no override needed -- FitConfig's own default
is now wide enough). Compare against results_nbins_sweep.json's existing
nbins=89 rows (generated with the OLD narrow grid) -- final xlam should
converge to the same value either way (bisection root is independent of
the initial bracket, confirmed directly this session), just found faster.
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
        # xlam_auto_grid left at FitConfig's own (now-widened) default
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
    nbins = 89
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

    with open("dev_notes/stress_test/results_nbins89_sweep_recalibrated.json", "w") as f:
        json.dump(results, f, indent=2)
