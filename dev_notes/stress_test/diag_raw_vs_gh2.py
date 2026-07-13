"""Check whether the GH-moment fit adds V bias beyond the raw histogram's
own first moment -- using the VALIDATED wide (8000-9000A) window this time
(diag_raw_vs_gh.py accidentally used harness.py's stale narrow 8415-8750A
window via fit_one_emiles's hardcoded import, invalidating that run)."""
import sys
sys.path.insert(0, "dev_notes/stress_test")
from harness_emiles import build_emiles_mock, write_spec_file, WAVEMIN
from kinextract import FitConfig, run_spectral_fit, set_verbose
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite
from kinextract.errors import _losvd_moments
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
    b = fit["outputs"]["b"]
    gh = fit_losvd_gauss_hermite(fit["state"].xl, b, fit_h3h4=True)
    mv, ms = _losvd_moments(fit["state"].xl, b)
    return dict(gh_v=gh["vherm"], gh_sigma=gh["sherm"], raw_v=mv, raw_sigma=ms)

if __name__ == "__main__":
    sigmas = [30, 70, 160, 250, 350]
    seeds = [42, 1]
    true_v = 80.0
    tmpdir = tempfile.mkdtemp()
    print(f"{'sigma':>6s} {'seed':>5s} {'gh_v':>8s} {'raw_v':>8s} {'gh_sigma':>9s} {'raw_sigma':>10s}")
    for sigma in sigmas:
        for seed in seeds:
            gal, errs = build_emiles_mock(true_v, sigma, seed)
            spec_path = f"{tmpdir}/mock_{sigma}_{seed}.spec"
            write_spec_file(gal, errs, spec_path)
            r = fit_kinextract(spec_path, sigma)
            print(f"{sigma:6.0f} {seed:5d} {r['gh_v']:8.2f} {r['raw_v']:8.2f} {r['gh_sigma']:9.2f} {r['raw_sigma']:10.2f}")
