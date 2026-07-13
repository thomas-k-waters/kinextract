"""Diagnose whether estimate_velocity_xcorr's V estimate (which sets the
wing-taper's v_center pivot) is itself biased at high sigma -- a candidate
explanation for the residual V undershoot that persists regardless of bin
count or xlam_criterion."""
import sys, time
sys.path.insert(0, "dev_notes/stress_test")
from harness_emiles import build_emiles_mock, write_spec_file, WAVEMIN
from kinextract import FitConfig, run_spectral_fit, set_verbose
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite
import kinextract.joint as jt
import numpy as np
import tempfile

EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"

orig_xcorr = jt.estimate_velocity_xcorr
calls = []
def spy(st, detrend_deg=3):
    v = orig_xcorr(st, detrend_deg)
    calls.append(v)
    return v
jt.estimate_velocity_xcorr = spy

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
    v_center_final = getattr(fit["state"], "v_center", None)
    return dict(v=gh["vherm"], sigma_rec=gh["sherm"], v_center_final=v_center_final)

if __name__ == "__main__":
    sigmas = [30, 70, 120, 160, 250, 350]
    true_v = 80.0
    tmpdir = tempfile.mkdtemp()
    print(f"{'sigma':>6s} {'xcorr_calls (v_center per sigl0 iter)':<50s} {'final_v_center':>14s} {'recovered_V':>12s}")
    for sigma in sigmas:
        gal, errs = build_emiles_mock(true_v, sigma, 42)
        spec_path = f"{tmpdir}/mock_{sigma}.spec"
        write_spec_file(gal, errs, spec_path)
        calls.clear()
        r = fit_kinextract(spec_path, sigma)
        calls_str = ", ".join(f"{c:.2f}" for c in calls)
        print(f"{sigma:6.0f} {calls_str:<50s} {r['v_center_final']:14.3f} {r['v']:12.3f}")
