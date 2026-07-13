"""Same sigma sweep as before, but with a mock built from a broad, smooth,
realistic population mixture spanning the full 20-SSP fitting grid (not
just 2 discrete SSPs) -- tests whether the earlier V bias was an artifact
of the too-simple 2-SSP mock's mismatch with the fitting library's
flexibility."""
import sys, time, json
sys.path.insert(0, "dev_notes/stress_test")
from harness_emiles_broad import build_broad_mock
from harness import write_spec_file, WAVEMIN
from kinextract import FitConfig, run_spectral_fit, set_verbose
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite
import numpy as np
import tempfile

EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"

def fit_kinextract(spec_path, sigma_guess):
    vrange = max(300.0, 3.5 * sigma_guess)
    cfg = FitConfig(
        template_list_file=EMILES_TLIST, template_dir=EMILES_DIR,
        wavemin_full=WAVEMIN, step=1.25, wavefitmin=8000.0, wavefitmax=9000.0,
        zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
        losvd_vmin=-vrange, losvd_vmax=vrange, sigl=sigma_guess, clean=False,
        map_maxiter=10000, print_every=999999,
    )
    fit = run_spectral_fit(cfg, gal_file=str(spec_path))
    gh = fit_losvd_gauss_hermite(fit["state"].xl, fit["outputs"]["b"], fit_h3h4=True)
    return dict(v=gh["vherm"], sigma_rec=gh["sherm"], chi2_red=fit["outputs"]["chi2_red"])

if __name__ == "__main__":
    sigmas = [30, 70, 120, 160, 250, 300, 350]
    seeds = [42, 1, 7]
    true_v = 80.0
    tmpdir = tempfile.mkdtemp()
    results = []
    t0 = time.time()
    print(f"{'sigma':>6s} {'seed':>5s} {'V':>8s} {'sigma_rec':>9s} {'chi2_red':>9s}")
    for sigma in sigmas:
        for seed in seeds:
            gal, errs = build_broad_mock(true_v, sigma, seed)
            spec_path = f"{tmpdir}/mock_broad_{sigma}_{seed}.spec"
            write_spec_file(gal, errs, spec_path)
            r = fit_kinextract(spec_path, sigma)
            results.append(dict(sigma=sigma, seed=seed, true_v=true_v, **r))
            print(f"{sigma:6.0f} {seed:5d} {r['v']:8.2f} {r['sigma_rec']:9.2f} {r['chi2_red']:9.3f}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"\n{'sigma':>6s} {'Vbias':>8s} {'Vstd':>7s} {'sigbias':>8s} {'sigstd':>7s}")
    for sigma in sigmas:
        rows = [r for r in results if r["sigma"] == sigma]
        vb = np.array([r["v"] for r in rows]) - true_v
        sb = np.array([r["sigma_rec"] for r in rows]) - sigma
        print(f"{sigma:6.0f} {vb.mean():+8.2f} {vb.std():7.2f} {sb.mean():+8.2f} {sb.std():7.2f}")

    with open("dev_notes/stress_test/results_broad_mock_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
