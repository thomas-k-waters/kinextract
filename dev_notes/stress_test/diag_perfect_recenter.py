"""Force the wing-taper's v_center pivot to the EXACT true V (bypassing the
real xcorr estimate) and see if the shrinkage-toward-zero effect (found in
diag_v_offset.py: bias grows with |true_v|, flips sign with sign(true_v))
disappears. Isolates "grid geometry / imperfect recentering" from "genuinely
symmetric grid effect independent of recentering quality"."""
import sys
sys.path.insert(0, "dev_notes/stress_test")
from harness_emiles import build_emiles_mock, write_spec_file, WAVEMIN
from kinextract import FitConfig, run_spectral_fit, set_verbose
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite
import kinextract.joint as jt
import tempfile

EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"

def fit_kinextract(spec_path, sigma_guess, true_v_for_patch=None):
    vrange = max(300.0, 3.5 * sigma_guess)
    cfg = FitConfig(
        template_list_file=EMILES_TLIST, template_dir=EMILES_DIR,
        wavemin_full=WAVEMIN, step=1.25, wavefitmin=8000.0, wavefitmax=9000.0,
        zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
        losvd_vmin=-vrange, losvd_vmax=vrange, sigl=sigma_guess, clean=False,
        map_maxiter=10000, print_every=999999,
    )
    if true_v_for_patch is not None:
        orig = jt.estimate_velocity_xcorr
        jt.estimate_velocity_xcorr = lambda st, detrend_deg=3: true_v_for_patch
        try:
            fit = run_spectral_fit(cfg, gal_file=str(spec_path))
        finally:
            jt.estimate_velocity_xcorr = orig
    else:
        fit = run_spectral_fit(cfg, gal_file=str(spec_path))
    gh = fit_losvd_gauss_hermite(fit["state"].xl, fit["outputs"]["b"], fit_h3h4=True)
    return dict(v=gh["vherm"], sigma_rec=gh["sherm"])

if __name__ == "__main__":
    sigma = 70
    seeds = [42, 1, 7]
    tmpdir = tempfile.mkdtemp()
    print(f"{'true_v':>7s} {'seed':>5s} {'mode':>12s} {'V':>8s} {'bias':>8s}")
    for true_v in [40.0, 80.0, -80.0]:
        for seed in seeds:
            gal, errs = build_emiles_mock(true_v, sigma, seed)
            spec_path = f"{tmpdir}/mock_v{true_v}_{seed}.spec"
            write_spec_file(gal, errs, spec_path)
            r_normal = fit_kinextract(spec_path, sigma, true_v_for_patch=None)
            r_perfect = fit_kinextract(spec_path, sigma, true_v_for_patch=true_v)
            print(f"{true_v:7.1f} {seed:5d} {'xcorr(normal)':>12s} {r_normal['v']:8.2f} {r_normal['v']-true_v:+8.2f}")
            print(f"{true_v:7.1f} {seed:5d} {'perfect':>12s} {r_perfect['v']:8.2f} {r_perfect['v']-true_v:+8.2f}")
