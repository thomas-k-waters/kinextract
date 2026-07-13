"""Does reducing the fitting template count (closer to the true 2-SSP
generating population) reduce the V bias at high sigma? Tests the
hypothesis that template-weight/LOSVD-shift degeneracy (more templates =
more freedom to trade weights against a shifted/narrower LOSVD to fit noise)
is the driver of the residual V bias found by diag_true_init.py."""
import sys, time
sys.path.insert(0, "dev_notes/stress_test")
from harness_emiles import build_emiles_mock, write_spec_file, WAVEMIN
from kinextract import FitConfig, run_spectral_fit, set_verbose
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite
import numpy as np
import tempfile

TEMPLATE_SETS = {
    "full20": ("dev_notes/stress_test/emiles_templates/Tlist", "dev_notes/stress_test/emiles_templates"),
    "med16":  ("dev_notes/stress_test/emiles_templates_small/Tlist", "dev_notes/stress_test/emiles_templates_small"),
    "tiny4":  ("dev_notes/stress_test/emiles_templates_tiny/Tlist", "dev_notes/stress_test/emiles_templates_tiny"),
}

def fit_kinextract(spec_path, sigma_guess, tlist, tdir):
    vrange = max(300.0, 3.5 * sigma_guess)
    cfg = FitConfig(
        template_list_file=tlist, template_dir=tdir,
        wavemin_full=WAVEMIN, step=1.25, wavefitmin=8000.0, wavefitmax=9000.0,
        zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
        losvd_vmin=-vrange, losvd_vmax=vrange, sigl=sigma_guess, clean=False,
        map_maxiter=10000, print_every=999999,
    )
    fit = run_spectral_fit(cfg, gal_file=str(spec_path))
    gh = fit_losvd_gauss_hermite(fit["state"].xl, fit["outputs"]["b"], fit_h3h4=True)
    return dict(v=gh["vherm"], sigma_rec=gh["sherm"], chi2_red=fit["outputs"]["chi2_red"])

if __name__ == "__main__":
    sigmas = [70, 250, 300, 350]
    seeds = [42, 1, 7]
    true_v = 80.0
    tmpdir = tempfile.mkdtemp()
    print(f"{'sigma':>6s} {'seed':>5s} {'set':>8s} {'V':>8s} {'sigma_rec':>9s} {'chi2_red':>9s}")
    results = []
    for sigma in sigmas:
        for seed in seeds:
            gal, errs = build_emiles_mock(true_v, sigma, seed)
            spec_path = f"{tmpdir}/mock_{sigma}_{seed}.spec"
            write_spec_file(gal, errs, spec_path)
            for name, (tlist, tdir) in TEMPLATE_SETS.items():
                r = fit_kinextract(spec_path, sigma, tlist, tdir)
                results.append(dict(sigma=sigma, seed=seed, set=name, true_v=true_v, **r))
                print(f"{sigma:6.0f} {seed:5d} {name:>8s} {r['v']:8.2f} {r['sigma_rec']:9.2f} {r['chi2_red']:9.3f}")

    print(f"\n{'sigma':>6s} {'set':>8s} {'Vbias':>8s} {'Vstd':>7s} {'sigbias':>8s} {'sigstd':>7s}")
    for sigma in sigmas:
        for name in TEMPLATE_SETS:
            rows = [r for r in results if r["sigma"] == sigma and r["set"] == name]
            vb = np.array([r["v"] for r in rows]) - true_v
            sb = np.array([r["sigma_rec"] for r in rows]) - sigma
            print(f"{sigma:6.0f} {name:>8s} {vb.mean():+8.2f} {vb.std():7.2f} {sb.mean():+8.2f} {sb.std():7.2f}")
