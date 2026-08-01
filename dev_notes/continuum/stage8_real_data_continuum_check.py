"""VALIDATION_CAMPAIGN.md continuum stage: the user's own final test
criterion -- "get a consistent answer as the real data on the
pre-normalized spectra (accounting for error introduced in the
pre-normalization, which definitely wasn't perfect)." Uses the RAW,
non-pre-normalized .dat templates (not the .txt continuum-normalized
ones fit_all_bins.py uses) -- the whole point of the continuum model is
to not need pre-normalized templates. Each raw template is divided by
its own median flux first (a harmless overall-amplitude rescale to O(1),
NOT a continuum-shape normalization -- NNLS weights are amplitude-
agnostic anyway) purely so the continuum spline's C~1 starting guess is
sensible; this is not the library-keyed "template continuum trust"
mechanism from the architecture plan (not implemented yet).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from kinextract import (FitConfig, build_template_stack, compute_losvd_moments,
                              build_bspline_basis, difference_matrix, fit_losvd_with_continuum)

DATA_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/normalizing_flows/data/N5102/kinematics/muse/kinematic_extraction"
RAW_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates"
Z = 0.0016181
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7

with open(os.path.join(DATA_DIR, "pallmc.out")) as f:
    rows = [l.split() for l in f if l.strip()]
pallmc = {r[9].replace("spmc.sim", ""): dict(radius=float(r[0]), V=float(r[1]), Verr=float(r[2]),
          sig=float(r[3]), sigerr=float(r[4]), h4=float(r[7])) for r in rows}

with open(os.path.join(RAW_TEMPLATE_DIR, "Tlist")) as f:
    tnames = [line.strip() for line in f if line.strip() and line.strip().endswith(".dat")]
raw = [np.loadtxt(os.path.join(RAW_TEMPLATE_DIR, name)) for name in tnames]
template_wave = raw[0][:, 0]
template_fluxes = [np.interp(template_wave, td[:, 0], td[:, 1]) for td in raw]
# harmless overall-amplitude rescale (see module docstring) -- NOT continuum-shape normalization
template_fluxes = [t / np.median(t) for t in template_fluxes]
print(f"loaded {len(template_fluxes)} RAW (non-pre-normalized) templates")

SUBSET = ["bin0105", "bin0504", "bin0601", "bin0704", "bin0902", "bin1005",
          "bin1401", "bnn1603", "bnn1604", "bnn1605"]

t0 = time.time()
for binname in SUBSET:
    b = pallmc[binname]
    d = np.loadtxt(os.path.join(DATA_DIR, f"{binname}sp.norm"))
    wave_obs, flux_obs = d[:, 0], d[:, 1]
    wave_rest = wave_obs / (1.0 + Z)
    mask = (wave_rest >= WAVEFITMIN) & (wave_rest <= WAVEFITMAX)
    wave_fit, flux_fit = wave_rest[mask], flux_obs[mask]
    noise_level = float(np.loadtxt(os.path.join(DATA_DIR, f"{binname}sp.rms"))[0])

    vmin, vmax = b["V"] - NSIG * b["sig"], b["V"] + NSIG * b["sig"]
    cfg = FitConfig(vmin=vmin, vmax=vmax)
    v_grid = cfg.build_velocity_grid()
    A_stack = build_template_stack(wave_fit, template_wave, template_fluxes, v_grid)
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = flux_fit / errors

    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D2 = difference_matrix(n_coef, order=2, dv=dv, dv_ref=cfg.knot_spacing_kms)

    result = fit_losvd_with_continuum(A_stack_w, y_w, wave_fit, basis, D2, n_continuum_knots=6, lam=cfg.lam)
    m = compute_losvd_moments(v_grid, result["b"])

    print(f"{binname:10s} r={b['radius']:+7.2f}  n_iter={result['n_iter']:2d}  success={result['success']}  "
          f"V={m['vherm']:+7.2f} (legacy {b['V']:+7.2f})  "
          f"sig={m['sherm']:6.2f} (legacy {b['sig']:6.2f})  h4={m['h4']:+.3f} (legacy {b['h4']:+.3f})  "
          f"t={time.time()-t0:.0f}s", flush=True)

print(f"\ntime: {time.time()-t0:.0f}s")
