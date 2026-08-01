"""Builds a regularization-bias calibration table with a real `m_eff`
axis -- the extension `calibrate_bias_muse_multitemplate.py`'s shipped
table doesn't have, needed before the Pinkney-style correction can be
validly applied to a fit made with *adaptive* regularization
(`select_lambda_aic_local`) rather than this package's fixed `lam=2.0`
default (see `bias_correction.correct_moments`'s own `m_eff` parameter,
and this session's Stage-C coupling discussion for why the two can't
just be mixed using the old table as-is).

**Same real MUSE templates, same fixed mixture, same (sigma, S/N) grid**
as the shipped table (`calibrate_bias_muse_multitemplate.py`), so the two
are directly comparable -- the only addition is `lam_grid` (the same grid
`select_lambda_aic_local` itself scans), giving each (sigma, S/N) cell
`len(lam_grid)` calibration points instead of one, each with its own
measured `m_eff`.

**Parallelized across (sigma, S/N) cells** via `ProcessPoolExecutor` --
8x more fits than the original table (one per lam value instead of one
fixed lam), so this needed the same module-level-worker-function
treatment as `bias_sweep_fixed_vs_adaptive.py` (a notebook-local function
can't be pickled for a spawned worker process on macOS; this reuses the
existing, already-tested `calibrate_bias.run_calibration_sweep` as the
per-cell worker, called once per cell with a single-element sigma/snr
grid -- verified equivalent to one big sequential call, since
`run_calibration_sweep`'s own per-replicate seed only depends on that
cell's own (sigma, snr, rep), never on what else is in the same sweep).

Usage: python calibrate_bias_muse_multitemplate_meff.py [outfile]
"""
import sys, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import run_calibration_sweep, save_calibration_table

TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WEIGHT_SEED = 10000  # same fixed Dirichlet mixture as the shipped table, for direct comparability

SIGMA_GRID = [20, 30, 50, 80, 140, 220, 300, 400]
SNR_GRID = [10, 15, 25, 40, 60, 100, 150, 200]
N_REPS = 20
LAM_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # same grid select_lambda_aic_local scans


def _load_templates_and_weights():
    with open(os.path.join(TEMPLATE_DIR, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    wavelength = 4750.0 + np.arange(3681) * 1.25
    templates = [np.interp(wavelength, *np.loadtxt(os.path.join(TEMPLATE_DIR, n)).T) for n in names]
    rng = np.random.default_rng(WEIGHT_SEED)
    weights = rng.dirichlet(np.ones(len(templates)))
    return templates, wavelength, weights


def _run_one_cell(sigma, snr):
    """One (sigma, S/N) cell, all `len(LAM_GRID)` lam values, `N_REPS`
    replicates each -- module-level so `ProcessPoolExecutor` can pickle a
    reference to it. Reloads templates/weights per call (cheap relative
    to the actual fitting cost, and avoids pickling large arrays through
    the process pool for every task).
    """
    templates, wavelength, weights = _load_templates_and_weights()
    rows = run_calibration_sweep(
        templates, wavelength, weights,
        sigma_grid=[sigma], snr_grid=[snr], n_reps=N_REPS,
        lam_grid=LAM_GRID, verbose=False,
    )
    return rows


def run(sigma_grid=SIGMA_GRID, snr_grid=SNR_GRID, max_workers=None, verbose=True):
    max_workers = max_workers or os.cpu_count()
    cells = [(sigma, snr) for sigma in sigma_grid for snr in snr_grid]
    all_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one_cell, sigma, snr): (sigma, snr) for sigma, snr in cells}
        n_done = 0
        for fut in as_completed(futures):
            sigma, snr = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)
            n_done += 1
            if verbose:
                print(f"  cell (sigma={sigma}, snr={snr}) done -- {len(rows)} lam rows -- "
                      f"{n_done}/{len(cells)} cells, t={time.time()-t0:.0f}s", flush=True)
    print(f"total wall time: {time.time()-t0:.0f}s ({max_workers} workers), {len(all_rows)} total rows", flush=True)
    return all_rows


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "kinextract", "calibration_data",
        "regularization_bias_muse_multitemplate_meff.txt",
    )
    templates, _, _ = _load_templates_and_weights()
    print(f"loaded {len(templates)} real MUSE templates, {len(SIGMA_GRID)}x{len(SNR_GRID)} grid, "
          f"{len(LAM_GRID)} lam values, n_reps={N_REPS} -- "
          f"{len(SIGMA_GRID)*len(SNR_GRID)*len(LAM_GRID)*N_REPS} total fits")
    rows = run()
    save_calibration_table(
        rows, outfile,
        header_comment=(
            f"Real MUSE templates ({len(templates)}), fixed Dirichlet mixture (seed={WEIGHT_SEED}), "
            f"V=0, n_reps={N_REPS}, lam_grid={LAM_GRID} -- m_eff axis for use with adaptive "
            f"regularization (select_lambda_aic_local). Same (sigma, S/N) grid as "
            f"regularization_bias_muse_multitemplate.txt for direct comparability."
        ),
    )
    print(f"wrote {len(rows)} rows to {outfile}")
