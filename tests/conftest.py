"""Shared fixtures for kinextract tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the src layout importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Common grid parameters ──────────────────────────────────────────────────

N_VEL = 101        # LOSVD velocity bins
V_MAX = 600.0      # km/s half-range
N_PIX = 800        # wavelength pixels
N_TMPL = 3         # number of templates
RNG_SEED = 42


@pytest.fixture(scope="session")
def velocity_grid():
    """Symmetric velocity grid from -V_MAX to +V_MAX."""
    return np.linspace(-V_MAX, V_MAX, N_VEL)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def wavelength_grid():
    """Log-spaced wavelength grid similar to CaII triplet region."""
    return np.exp(np.linspace(np.log(8400.0), np.log(8800.0), N_PIX))


@pytest.fixture(scope="session")
def gaussian_losvd(velocity_grid):
    """Unit-normalised Gaussian LOSVD with sigma=120 km/s, mean=50 km/s."""
    v = velocity_grid
    mean, sigma = 50.0, 120.0
    b = np.exp(-0.5 * ((v - mean) / sigma) ** 2)
    b /= b.sum()
    return b


@pytest.fixture(scope="session")
def mock_templates(rng, wavelength_grid):
    """N_TMPL random smooth templates on the wavelength grid."""
    templates = np.ones((N_TMPL, N_PIX))
    for i in range(N_TMPL):
        # Smooth random spectrum: low-frequency variations + a couple of dips
        x = np.linspace(0, 4 * np.pi, N_PIX)
        templates[i] = 1.0 + 0.3 * np.sin((i + 1) * x) - 0.1 * np.cos(2 * x)
        templates[i] = np.clip(templates[i], 0.01, None)
    return templates


@pytest.fixture(scope="session")
def real_muse_fit():
    """A real MAP fit on the bundled, pre-normalized MUSE example spectrum.

    Uses the real ``.norm`` file (produced by the user's own Greatlakes
    pipeline) rather than a raw ``.spec`` file, so this exercises the
    genuinely non-joint, pre-normalized-mode (``fit_continuum=False``) MAP
    path -- the shipped ``a_map`` parameter layout that
    :mod:`kinextract.errors`/:mod:`kinextract.validation`'s
    Laplace/bias-correction/recovery-bias machinery requires (all of which
    refuse continuum-cofit/joint-mode fits). Skipped if the bundled example
    data isn't present (e.g. a minimal install without the examples/
    directory). Session-scoped since it's an expensive real fit, reused
    across any test that needs a realistic (rather than synthetic)
    FitState/result pair.
    """
    import tempfile
    from kinextract import FitConfig, run_spectral_fit

    muse_dir = Path(__file__).parent.parent / "examples" / "data" / "muse"
    norm_file = muse_dir / "bin0105sp.norm"
    if not norm_file.exists():
        pytest.skip("bundled MUSE example data not found")

    outdir = tempfile.mkdtemp(prefix="kinextract_test_muse_")

    cfg = FitConfig(
        template_list_file=str(muse_dir / "Tlist"),
        template_dir=str(muse_dir),
        outdir=outdir,
        step=1.25,
        wavefitmin=8400.0, wavefitmax=8750.0,
        zgal=0.001556,
        losvd_vmin=-300.0, losvd_vmax=300.0,
        fit_continuum=False,
        xlam_auto=True, xlam_criterion="roughness", xlam_smooth_threshold=0.25,
        sigl=100.0, clean=True,
        map_maxiter=20000, print_every=0,
    )
    fit = run_spectral_fit(cfg, gal_file=str(norm_file))
    return fit, cfg
