"""Is the persistent ~-2 km/s low-sigma V bias a fixed additive offset
(present regardless of the true V), or does it scale with true V? Test at
true_v=0 and true_v=80, both at a well-behaved low sigma."""
import sys
sys.path.insert(0, "dev_notes/stress_test")
from harness_emiles import build_emiles_mock, write_spec_file, WAVEMIN
from kinextract import FitConfig, run_spectral_fit, set_verbose
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite
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
    return dict(v=gh["vherm"], sigma_rec=gh["sherm"])

if __name__ == "__main__":
    sigma = 70
    seeds = [42, 1, 7]
    tmpdir = tempfile.mkdtemp()
    print(f"{'true_v':>7s} {'seed':>5s} {'V':>8s} {'bias':>8s}")
    for true_v in [0.0, 40.0, 80.0, -80.0]:
        for seed in seeds:
            gal, errs = build_emiles_mock(true_v, sigma, seed)
            spec_path = f"{tmpdir}/mock_v{true_v}_{seed}.spec"
            write_spec_file(gal, errs, spec_path)
            r = fit_kinextract(spec_path, sigma)
            print(f"{true_v:7.1f} {seed:5d} {r['v']:8.2f} {r['v']-true_v:+8.2f}")
