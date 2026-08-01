"""Does the too-narrow bootstrap band (measured elsewhere: ~2x too narrow at
sigma=50, ~4x at sigma=90) come partly from bootstrap_losvd pinning lam at
the point estimate's own selection for every replicate?

Confirmed by direct code read (uncertainty.py bootstrap_losvd): every
replicate is fit with `cfg_boot = dataclasses.replace(cfg, lam=result0[
"lam_used"], lam_selection="fixed", ...)` -- lam is NEVER reselected. Given
lam selection is a documented lottery at S/N 60-150 (flips between e.g. 8 and
32768 across noise seeds on identical truth), this pins the bootstrap to
whatever regularization regime the point estimate happened to land in, and
structurally cannot explore "what if this noise draw had pushed selection
somewhere else."

Three ensembles, same mock, same n=30:
  A. current bootstrap_losvd as shipped (lam pinned from point estimate)
  B. same bootstrap MECHANISM (noise injected around the point estimate's
     own model) but lam RESELECTED per replicate (cfg unmodified, adaptive)
  C. the actual gold standard: independent noise draws around the TRUE
     model (not the point estimate's), each fit fresh with full adaptive
     selection -- this is literally "repeat the real observation N times",
     the quantity a bootstrap is trying to approximate.

If B's spread is close to C's and much larger than A's, the lam pin is a
real, fixable contributor to the under-coverage. If B ~= A, the pin isn't
where the missing variance lives and the fix has to be found elsewhere.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import dataclasses
import numpy as np

from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                             compute_losvd_moments, load_emiles_templates, whiten)
from kinextract.multi_template import fit_losvd
from kinextract.uncertainty import bootstrap_losvd

EM = "src/kinextract/template_data/emiles_muse_grid.npz"
SEL = [(1.0, -0.66), (5.0, 0.06), (14.0, 0.06), (14.0, 0.40)]
TW = np.array([.15, .35, .35, .15])
N_ENSEMBLE = 30


def build(sigma, snr, V=30.0, h3=0.0, h4=0.0):
    T, tw, *_ = load_emiles_templates(EM, select=SEL)
    obs = 4750. + np.arange(3681) * 1.25
    wave = obs[(obs >= 8400) & (obs <= 8800)]
    cfg = FitConfig(vmin=V - 7 * sigma, vmax=V + 7 * sigma)
    vg = cfg.build_velocity_grid()
    bt = np.maximum(gauss_hermite_model(vg, 1., V, sigma, h3, h4), 0.); bt /= bt.sum()
    A = build_template_stack(wave, tw, T, vg)
    g0 = sum(TW[k] * (A[k] @ bt) for k in range(4))
    return cfg, vg, A, g0, wave


def reselect_bootstrap(A_stack_w, y_w, cfg, n_boot, seed, wave_fit=None,
                        force_single_start=True):
    """Same mechanism as bootstrap_losvd (noise injected around the point
    estimate's own model), but lam RESELECTED (adaptive) every replicate
    instead of pinned. `force_single_start=True` matches
    bootstrap_losvd's own default (replicate_single_start=True) so this
    isolates ONLY the lam-pin-vs-reselect difference -- an earlier version
    of this function left cfg's own multi-start setting untouched, which
    confounded the comparison with a start-strategy difference as well.
    Deliberately NOT touching the package -- this is a diagnostic copy, not
    a proposed replacement, until the comparison below says it should be."""
    rng = np.random.default_rng(seed)
    result0 = fit_losvd(A_stack_w, y_w, cfg, wave_fit=wave_fit)
    v_grid = cfg.build_velocity_grid()
    A_eff0 = np.tensordot(result0["w"], A_stack_w, axes=([0], [0]))
    model_w = A_eff0 @ result0["b"]
    n_pix = A_stack_w.shape[1]
    cfg_reselect = (dataclasses.replace(cfg, losvd_single_start=True)
                    if force_single_start else cfg)
    sherm = np.empty(n_boot); h4 = np.empty(n_boot); lams = np.empty(n_boot)
    for i in range(n_boot):
        y_w_boot = model_w + rng.standard_normal(n_pix)
        r = fit_losvd(A_stack_w, y_w_boot, cfg_reselect, wave_fit=wave_fit)  # adaptive, single-start
        m = compute_losvd_moments(v_grid, r["b"], fit_h3h4=True)
        sherm[i] = m["sherm"]; h4[i] = m["h4"]; lams[i] = r["lam_used"]
    return sherm, h4, lams


def true_ensemble(A, g0, wave, cfg, snr, n, seed):
    """Independent noise draws around the TRUE (not point-estimate) model,
    each fit fresh with full adaptive selection. The actual sampling
    distribution of the estimator under repeated real observation."""
    vg = cfg.build_velocity_grid()
    nl = np.median(g0) / snr
    sherm = np.empty(n); h4 = np.empty(n); lams = np.empty(n)
    for i in range(n):
        rng = np.random.default_rng(seed * 1000 + i)
        gal = g0 + rng.normal(0, nl, len(wave))
        A_w, y_w, _ = whiten(A, gal, np.full(len(wave), nl), rho=0.0)
        r = fit_losvd(A_w, y_w, cfg)
        m = compute_losvd_moments(vg, r["b"], fit_h3h4=True)
        sherm[i] = m["sherm"]; h4[i] = m["h4"]; lams[i] = r["lam_used"]
    return sherm, h4, lams


if __name__ == "__main__":
    print(f"{'regime':>18s} {'ensemble':>22s} {'sherm std':>10s} {'h4 std':>8s}  lam values", flush=True)
    for sigma, snr, label in [(50., 60., "sigma=50,S/N=60"), (90., 60., "sigma=90,S/N=60")]:
        cfg, vg, A, g0, wave = build(sigma, snr)
        nl = np.median(g0) / snr
        rng0 = np.random.default_rng(0)
        gal0 = g0 + rng0.normal(0, nl, len(wave))
        A_w, y_w, _ = whiten(A, gal0, np.full(len(wave), nl), rho=0.0)

        t0 = time.time()
        boot_A = bootstrap_losvd(A_w, y_w, cfg, n_boot=N_ENSEMBLE, seed=1)
        print(f"{label:>18s} {'A: pinned-lam boot':>22s} {boot_A['sherm_std']:10.3f} "
              f"{boot_A['h4_std']:8.4f}  lam_used(point est)={boot_A['result0']['lam_used']:g}"
              f"   [{time.time()-t0:.0f}s]")

        t0 = time.time()
        sherm_B, h4_B, lams_B = reselect_bootstrap(A_w, y_w, cfg, N_ENSEMBLE, seed=2)
        print(f"{label:>18s} {'B: reselect-lam boot':>22s} {sherm_B.std(ddof=1):10.3f} "
              f"{h4_B.std(ddof=1):8.4f}  lam range=[{lams_B.min():g},{lams_B.max():g}] "
              f"unique={sorted(set(lams_B.tolist()))}   [{time.time()-t0:.0f}s]")

        t0 = time.time()
        sherm_C, h4_C, lams_C = true_ensemble(A, g0, wave, cfg, snr, N_ENSEMBLE, seed=3)
        print(f"{label:>18s} {'C: TRUE sampling dist':>22s} {sherm_C.std(ddof=1):10.3f} "
              f"{h4_C.std(ddof=1):8.4f}  lam range=[{lams_C.min():g},{lams_C.max():g}] "
              f"unique={sorted(set(lams_C.tolist()))}   [{time.time()-t0:.0f}s]")
        print()
