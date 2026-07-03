"""Mock-spectrum generation for the kinextract overnight validation suite.

Builds synthetic galaxy/cluster spectra with known ground-truth kinematics
(V, sigma, h3, h4, or a double-peaked LOSVD) by convolving a real or
synthetic stellar template with an arbitrary LOSVD, optionally adding a
realistic continuum SED or emission-line contamination, then S/N-scaled
noise. Factors out the mock-generation approach already used in
examples/notebooks/01_basic_mock_fit.ipynb and 02_realistic_mock_fit.ipynb
into reusable, independently-testable functions (see tests/test_mock_spectrum.py).

Truth-generation uses kinextract's own `gauss_hermite_losvd_model` (Fortran/
legacy H3/H4 convention) rather than an independent GH polynomial basis, so
injected truth and `fit_losvd_gauss_hermite`'s recovered moments are
guaranteed to be on the same convention -- otherwise a "biased" h3/h4
comparison could just be an artifact of two different polynomial
normalizations, not a real recovery error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

from kinextract.losvd import gauss_hermite_losvd_model
from kinextract._utils import CEE


# =============================================================================
# Instrument setups
# =============================================================================

@dataclass(frozen=True)
class InstrumentSetup:
    name: str
    wavemin_full: float
    step: float
    n_pix: int
    wavefitmin: float
    wavefitmax: float
    lam_center: float
    template_dir: Path
    default_template: str
    mismatch_template: Optional[str] = None  # None if instrument has only one template
    # Resolving power R = lambda / FWHM_LSF. None means no instrumental LSF is
    # modeled at all (the historical behavior for MUSE_CAII/STIS_SETUP: the
    # only broadening applied is the injected LOSVD itself). Only consumed by
    # build_mock_spectrum when include_instrument_lsf=True.
    resolving_power: Optional[float] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Gaussian FWHM = 2*sqrt(2*ln(2)) * sigma -- same convention as
# kinextract.templates.resolution_mismatch_sigma_A, so LSF sigmas computed
# here line up with FitConfig.data_fwhm_A/template_fwhm_A if a caller wants to
# exercise that real LSF-matching machinery against these mocks.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def instrument_lsf_sigma_kms(instrument: "InstrumentSetup") -> Optional[float]:
    """Gaussian LSF sigma (km/s) implied by ``instrument.resolving_power``.

    Returns None if the instrument has no ``resolving_power`` set (no LSF
    modeled). Used both to build the LSF convolution kernel in
    :func:`build_mock_spectrum` and to choose sigma_true test points relative
    to an instrument's own resolution limit.
    """
    if not instrument.resolving_power:
        return None
    fwhm_kms = CEE / instrument.resolving_power
    return fwhm_kms * _FWHM_TO_SIGMA


MUSE_CAII = InstrumentSetup(
    name="muse_caii", wavemin_full=4749.65, step=1.25, n_pix=3681,
    wavefitmin=8400.0, wavefitmax=8800.0, lam_center=8580.0,
    template_dir=_repo_root() / "examples" / "data" / "muse",
    default_template="HD102212_av.dat",
    mismatch_template="HD096446_av.dat",
    resolving_power=3000.0,  # MUSE WFM, R~3000 near CaII (~42 km/s LSF sigma)
)
STIS_SETUP = InstrumentSetup(
    name="stis", wavemin_full=8276.27, step=0.5547, n_pix=1024,
    wavefitmin=8350.0, wavefitmax=8800.0, lam_center=8575.0,
    template_dir=_repo_root() / "examples" / "data" / "stis",
    default_template="HR7615ecbig.dat",
    mismatch_template=None,
    resolving_power=5000.0,  # R~5000 (~25 km/s LSF sigma)
)

# Additional common optical spectrographs, all anchored at the same Ca II
# triplet window (8400-8800 A) as MUSE_CAII/STIS_SETUP so the same synthetic
# template, emission-line list, and masking logic apply unchanged -- only R
# (hence LSF width) and the native pixel scale differ, each set to a realistic
# published/representative value for that instrument (not tuned to make any
# particular test outcome come out a certain way). Reuse the bundled MUSE
# template as their "real" template source (interpolation onto a new pixel
# grid works regardless of the file's native sampling).
_MUSE_TEMPLATE_DIR = _repo_root() / "examples" / "data" / "muse"

SDSS_BOSS_SETUP = InstrumentSetup(
    name="sdss_boss", wavemin_full=8000.0, step=1.974, n_pix=700,
    wavefitmin=8400.0, wavefitmax=8800.0, lam_center=8580.0,
    template_dir=_MUSE_TEMPLATE_DIR, default_template="HD102212_av.dat",
    resolving_power=2000.0,  # BOSS spectrograph, ~69 km/s/pixel log-lambda dispersion
)
DEIMOS_SETUP = InstrumentSetup(
    name="deimos", wavemin_full=8000.0, step=0.33, n_pix=3700,
    wavefitmin=8400.0, wavefitmax=8800.0, lam_center=8580.0,
    template_dir=_MUSE_TEMPLATE_DIR, default_template="HD102212_av.dat",
    resolving_power=6500.0,  # Keck/DEIMOS, 1200 l/mm grating near the Ca II triplet
)
XSHOOTER_VIS_SETUP = InstrumentSetup(
    name="xshooter_vis", wavemin_full=8000.0, step=0.2, n_pix=6100,
    wavefitmin=8400.0, wavefitmax=8800.0, lam_center=8580.0,
    template_dir=_MUSE_TEMPLATE_DIR, default_template="HD102212_av.dat",
    resolving_power=8800.0,  # VLT/X-shooter VIS arm, ~1.0" slit
)

COMMON_SPECTROGRAPHS = (MUSE_CAII, STIS_SETUP, SDSS_BOSS_SETUP, DEIMOS_SETUP, XSHOOTER_VIS_SETUP)


# =============================================================================
# Templates
# =============================================================================

def load_and_normalize_template(path: Path, smooth_sigma_pix: float = 200.0
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a real bundled stellar template and continuum-normalize it.

    Reproduces the ``flux_raw / gaussian_filter(flux_raw, sigma=...)``
    idiom from 02_realistic_mock_fit.ipynb: divides by a heavily-smoothed
    version of itself so narrow absorption features survive but the broad
    SED shape is removed (continuum -> ~1).
    """
    data = np.loadtxt(path)
    wavelength, flux_raw = data[:, 0], data[:, 1]
    smooth_cont = gaussian_filter(flux_raw, sigma=smooth_sigma_pix)
    return wavelength, flux_raw, flux_raw / smooth_cont


def build_synthetic_caii_template(wavelength: np.ndarray,
                                   ca_centers=(8498.02, 8542.09, 8662.14),
                                   ca_depths=(0.55, 0.70, 0.65),
                                   ca_width_A: float = 5.0) -> np.ndarray:
    """From-scratch Gaussian-dip CaII template (01_basic_mock_fit.ipynb cell 3)."""
    template = np.ones_like(wavelength)
    for cen, depth in zip(ca_centers, ca_depths):
        template -= depth * np.exp(-0.5 * ((wavelength - cen) / ca_width_A) ** 2)
    return template


# =============================================================================
# LOSVD truth generation and convolution
# =============================================================================

def gauss_hermite_losvd_on_grid(v_grid: np.ndarray, v_true: float, sigma_true: float,
                                 h3: float = 0.0, h4: float = 0.0) -> np.ndarray:
    """Evaluate the true LOSVD on a fine velocity grid using kinextract's own
    `gauss_hermite_losvd_model` (amp=1), guaranteeing the same H3/H4
    convention as the recovery-side `fit_losvd_gauss_hermite`.
    """
    losvd = gauss_hermite_losvd_model(v_grid, amp=1.0, vel=v_true, sig=sigma_true, h3=h3, h4=h4)
    losvd = np.clip(losvd, 0.0, None)
    s = losvd.sum()
    return losvd / s if s > 0 else losvd


def double_peak_losvd_on_grid(v_grid: np.ndarray, v_center: float, sigma_true: float,
                               separation_frac: float = 0.6, width_frac: float = 0.5,
                               weight_ratio: float = 1.0) -> np.ndarray:
    """Two-Gaussian-component LOSVD: peaks at v_center +/- separation_frac*sigma_true,
    each with sigma = width_frac*sigma_true. weight_ratio = peak2/peak1 amplitude.
    """
    v1 = v_center - separation_frac * sigma_true
    v2 = v_center + separation_frac * sigma_true
    width = width_frac * sigma_true
    g1 = np.exp(-0.5 * ((v_grid - v1) / width) ** 2)
    g2 = np.exp(-0.5 * ((v_grid - v2) / width) ** 2)
    w2 = weight_ratio / (1.0 + weight_ratio)
    w1 = 1.0 - w2
    losvd = w1 * g1 + w2 * g2
    return losvd / losvd.sum()


def convolve_template_with_losvd(template: np.ndarray, step_A: float,
                                  v_grid: np.ndarray, losvd: np.ndarray,
                                  lam_center_A: float) -> np.ndarray:
    """General LOSVD convolution for an arbitrary (non-Gaussian/multi-peaked)
    LOSVD sampled on `v_grid`, via direct discrete convolution -- needed for
    the GH-moment and double-peak LOSVD-shape axes, where no closed-form
    scipy filter applies.

    For the pure-Gaussian case this must agree with the simpler
    `scipy.ndimage.gaussian_filter` + `shift` idiom (see
    :func:`apply_gaussian_losvd`) to within a small tolerance; this
    agreement is exactly what tests/test_mock_spectrum.py checks.
    """
    kernel_pix = v_grid * lam_center_A / (CEE * step_A)
    lo, hi = int(np.floor(kernel_pix.min())), int(np.ceil(kernel_pix.max()))
    pix_int = np.arange(lo, hi + 1)
    kernel_on_pix = np.interp(pix_int, kernel_pix, losvd, left=0.0, right=0.0)
    s = kernel_on_pix.sum()
    if s > 0:
        kernel_on_pix = kernel_on_pix / s
    return np.convolve(template, kernel_on_pix, mode="same")


def apply_gaussian_losvd(template: np.ndarray, v_true: float, sigma_true: float,
                          lam_center_A: float, step_A: float) -> np.ndarray:
    """Fast closed-form path for a pure Gaussian LOSVD (h3=h4=0), matching
    01_basic_mock_fit.ipynb/02_realistic_mock_fit.ipynb exactly.
    """
    sigma_pix = sigma_true * lam_center_A / (CEE * step_A)
    shift_pix = v_true * lam_center_A / (CEE * step_A)
    return ndimage_shift(gaussian_filter(template, sigma=sigma_pix), shift=+shift_pix)


# =============================================================================
# Continuum SED
# =============================================================================

def realistic_galaxy_continuum(wavelength: np.ndarray, cont_level: float = 12_000.0,
                                fringe_amplitude: float = 0.01) -> np.ndarray:
    """Cubic-polynomial + broad Gaussian hump (+ optional fringe) SED shape
    (02_realistic_mock_fit.ipynb cell 3). Used only for ALS-mode mocks,
    which expect a full-amplitude, un-normalized broadband continuum shape.
    """
    x_norm = (wavelength - wavelength.mean()) / (0.5 * (wavelength[-1] - wavelength[0]))
    slope = 1.0 + 0.80 * x_norm + 0.5 * x_norm ** 2 - 0.10 * x_norm ** 3
    hump = 0.50 * np.exp(-0.5 * ((x_norm - 0.20) / 0.45) ** 2)
    cont = cont_level * (slope + hump)
    if fringe_amplitude:
        fringes = fringe_amplitude * np.sin(2 * np.pi * (wavelength - wavelength[0]) / 50.0)
        cont = cont_level * (slope + hump + fringes)
    return cont


def mild_linear_tilt(wavelength: np.ndarray, slope_frac: float = 0.05) -> np.ndarray:
    """A gentle linear multiplicative tilt, continuum ~1 +/- slope_frac across
    the fit window. Used for poly_amp-mode mocks, so that mode is stress-tested
    inside the regime `continuum_poly_mode="multiplicative"` (a single linear
    coefficient, bound +/-0.1 by default) is actually designed for -- NOT the
    full factor-~4 ALS-oriented SED, which would be an unfair/uninformative
    comparison for a mode that was never meant to remove that much shape.
    """
    x_norm = (wavelength - wavelength.mean()) / (0.5 * (wavelength[-1] - wavelength[0]))
    return 1.0 + slope_frac * x_norm


# =============================================================================
# Emission-line contamination
# =============================================================================

# Rest-frame "em"-tagged lines from kinextract.continuum's default CaII-region
# line table (the same wavelengths build_emission_line_mask actually scans
# for), so injected contamination is representative of what the real
# detector looks for, not an arbitrary/unrecognized line.
DEFAULT_EMISSION_LINES_A = (8446.36, 8578.70, 8616.95, 8680.28, 8683.40, 8727.13, 9068.60)

EMISSION_AMPLITUDE_RANGE = {
    "none": (0.0, 0.0),
    "moderate": (0.3, 1.0),
    "strong": (1.0, 3.0),
}


def inject_emission_lines(wavelength: np.ndarray, flux: np.ndarray,
                           line_centers_A, continuum_level, amplitude_range: tuple[float, float],
                           fwhm_A: float = 3.0, rng: Optional[np.random.Generator] = None
                           ) -> np.ndarray:
    """Add Gaussian emission features at `line_centers_A` atop `flux`.

    Peak amplitude for each line is drawn uniformly from `amplitude_range`
    (interpreted as a multiple of `continuum_level`, scalar or per-pixel
    array) -- i.e. amplitude_range=(0.3, 1.0) means each line's peak height
    is 0.3-1.0x the local continuum.
    """
    if amplitude_range[1] <= 0:
        return flux.copy()
    rng = rng or np.random.default_rng(0)
    out = flux.copy()
    cont = np.broadcast_to(np.asarray(continuum_level, float), wavelength.shape)
    sigma_A = fwhm_A / 2.3548
    for cen in line_centers_A:
        if cen < wavelength.min() or cen > wavelength.max():
            continue
        local_cont = float(np.interp(cen, wavelength, cont))
        amp = rng.uniform(*amplitude_range) * local_cont
        out = out + amp * np.exp(-0.5 * ((wavelength - cen) / sigma_A) ** 2)
    return out


# =============================================================================
# Noise
# =============================================================================

def add_noise(flux: np.ndarray, snr_target: float, rng: np.random.Generator,
               noise_floor: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Add uniform per-pixel Gaussian noise at the given nominal S/N.

    Noise level is set from the median flux in the array passed in (caller
    should pass the fit-window slice, or accept a window-averaged S/N for
    the full spectrum) -- a single representative noise level, matching the
    "flat SNR" convention used in the real-data example notebooks, not a
    Poisson/flux-dependent model.
    """
    noise = max(float(np.median(np.abs(flux))) / snr_target, noise_floor)
    noisy = flux + rng.normal(0.0, noise, len(flux))
    err = np.full(len(flux), noise)
    return noisy, err


# =============================================================================
# .spec / template file I/O (format kinextract.io parses)
# =============================================================================

def write_spec_file(path: Path, flux: np.ndarray, err: np.ndarray) -> None:
    n = len(flux)
    np.savetxt(path, np.column_stack([np.arange(1, n + 1), flux, err]),
               fmt="%6d  %14.8f  %14.8f")


def write_template_file(path: Path, wavelength: np.ndarray, template: np.ndarray,
                         err_level: float = 0.001) -> None:
    np.savetxt(path, np.column_stack([wavelength, template, np.full(len(wavelength), err_level)]),
               fmt="%10.4f  %14.8f  %12.8f")


# =============================================================================
# Top-level mock spectrum container + builder
# =============================================================================

@dataclass
class MockSpectrum:
    wavelength: np.ndarray
    flux_noiseless: np.ndarray
    flux_noisy: np.ndarray
    errors: np.ndarray
    template_wavelength: np.ndarray
    template_flux: np.ndarray
    true_continuum: np.ndarray
    v_true: float
    sigma_true: float
    h3_true: float
    h4_true: float
    losvd_shape: str
    instrument: str
    noise_seed: int
    snr_target: float


def build_mock_spectrum(
    instrument: InstrumentSetup,
    v_true: float,
    sigma_true: float,
    snr_target: float,
    continuum_mode: Literal["als", "poly_amp", "none"],
    template_role: Literal["matched", "mismatched"],
    losvd_shape: Literal["gaussian", "gh_moderate", "gh_strong", "double_peak"],
    h3: float = 0.0,
    h4: float = 0.0,
    emission_level: Literal["none", "moderate", "strong"] = "none",
    noise_seed: int = 1000,
    use_real_template: bool = True,
    include_instrument_lsf: bool = False,
) -> MockSpectrum:
    """Top-level mock builder: dispatches to the helpers above according to
    the requested scenario axes and returns a fully-specified MockSpectrum.

    ``include_instrument_lsf`` (default False, preserving every existing
    scenario's exact prior behavior) adds a genuine instrumental
    line-spread-function convolution using ``instrument.resolving_power``;
    see the note above the LSF block below.
    """
    rng = np.random.default_rng(noise_seed)
    wavelength = instrument.wavemin_full + np.arange(instrument.n_pix) * instrument.step

    # --- Template: real (continuum-normalized) or synthetic ------------------
    if use_real_template:
        tmpl_name = instrument.default_template
        if template_role == "mismatched":
            if instrument.mismatch_template is None:
                raise ValueError(f"{instrument.name} has no mismatch_template; "
                                  "template_role='mismatched' is not available for this instrument.")
            gen_name = instrument.mismatch_template  # spectrum generated w/ this...
            fit_name = instrument.default_template   # ...but fit will use the default template
        else:
            gen_name = fit_name = tmpl_name
        gen_wave, _, gen_template_norm = load_and_normalize_template(instrument.template_dir / gen_name)
        template_for_generation = np.interp(wavelength, gen_wave, gen_template_norm, left=1.0, right=1.0)
        fit_wave, _, fit_template_norm = load_and_normalize_template(instrument.template_dir / fit_name)
        template_for_fitting = np.interp(wavelength, fit_wave, fit_template_norm, left=1.0, right=1.0)
    else:
        template_for_generation = template_for_fitting = build_synthetic_caii_template(wavelength)

    # --- Optional instrumental LSF (opt-in; off by default so existing
    # scenarios/results are unaffected by this addition) ----------------------
    # Blurs only the *generation* template, representing what a real detector
    # actually outputs; template_for_fitting is deliberately left sharp, so a
    # caller who sets FitConfig.data_fwhm_A/template_fwhm_A to match sees a
    # genuine data/template resolution mismatch resolved by kinextract's own
    # LSF-matching machinery (kinextract.templates.resolution_mismatch_sigma_A)
    # during the fit, rather than a mock that was silently pre-matched.
    if include_instrument_lsf and instrument.resolving_power:
        lsf_fwhm_A = instrument.lam_center / instrument.resolving_power
        lsf_sigma_pix = (lsf_fwhm_A * _FWHM_TO_SIGMA) / instrument.step
        template_for_generation = gaussian_filter(template_for_generation, sigma=lsf_sigma_pix)

    # --- LOSVD truth + broadening ---------------------------------------------
    # NOTE: the shape kernel is always built centered at v=0 (not v_true), and
    # the v_true shift is applied afterward via ndimage_shift. This matters:
    # np.convolve(..., mode="same") aligns the kernel's *array-index* center
    # with "zero lag" -- it has no notion of what physical velocity a given
    # array index represents. If the kernel were built centered at v_true
    # (nonzero), its probability mass would sit at the array's geometric
    # center for the WRONG reason (by construction, not because v_true=0),
    # and "same"-mode convolution would silently treat that as zero shift,
    # discarding the intended v_true offset. Building the shape at v=0 and
    # shifting explicitly afterward avoids this entirely, and matches
    # apply_gaussian_losvd's (already-correct) decomposition.
    if losvd_shape == "double_peak":
        v_grid = np.linspace(-6 * sigma_true, 6 * sigma_true, 4001)
        losvd = double_peak_losvd_on_grid(v_grid, 0.0, sigma_true)
        shape_broadened = convolve_template_with_losvd(template_for_generation, instrument.step,
                                                         v_grid, losvd, instrument.lam_center)
        shift_pix = v_true * instrument.lam_center / (CEE * instrument.step)
        broadened = ndimage_shift(shape_broadened, shift=+shift_pix)
    elif h3 != 0.0 or h4 != 0.0:
        v_grid = np.linspace(-8 * sigma_true, 8 * sigma_true, 4001)
        losvd = gauss_hermite_losvd_on_grid(v_grid, 0.0, sigma_true, h3=h3, h4=h4)
        shape_broadened = convolve_template_with_losvd(template_for_generation, instrument.step,
                                                         v_grid, losvd, instrument.lam_center)
        shift_pix = v_true * instrument.lam_center / (CEE * instrument.step)
        broadened = ndimage_shift(shape_broadened, shift=+shift_pix)
    else:
        broadened = apply_gaussian_losvd(template_for_generation, v_true, sigma_true,
                                          instrument.lam_center, instrument.step)

    # --- Continuum -------------------------------------------------------------
    if continuum_mode == "als":
        true_continuum = realistic_galaxy_continuum(wavelength, fringe_amplitude=0.01)
    elif continuum_mode == "poly_amp":
        true_continuum = mild_linear_tilt(wavelength) * float(np.median(np.abs(broadened))) * 1000.0
    else:  # "none" -- oracle/ceiling case, continuum is flat unity
        true_continuum = np.ones_like(wavelength)

    flux_noiseless = broadened * true_continuum

    # --- Emission contamination --------------------------------------------
    lo, hi = EMISSION_AMPLITUDE_RANGE[emission_level]
    if hi > 0:
        flux_noiseless = inject_emission_lines(
            wavelength, flux_noiseless, DEFAULT_EMISSION_LINES_A,
            true_continuum, (lo, hi), rng=rng,
        )

    # --- Noise (S/N measured over the fit window, not the full spectrum) ----
    fit_mask = (wavelength >= instrument.wavefitmin) & (wavelength <= instrument.wavefitmax)
    noise = max(float(np.median(np.abs(flux_noiseless[fit_mask]))) / snr_target, 1e-8)
    flux_noisy = flux_noiseless + rng.normal(0.0, noise, len(flux_noiseless))
    errors = np.full(len(flux_noiseless), noise)

    return MockSpectrum(
        wavelength=wavelength, flux_noiseless=flux_noiseless, flux_noisy=flux_noisy,
        errors=errors, template_wavelength=wavelength, template_flux=template_for_fitting,
        true_continuum=true_continuum, v_true=v_true, sigma_true=sigma_true,
        h3_true=h3, h4_true=h4, losvd_shape=losvd_shape, instrument=instrument.name,
        noise_seed=noise_seed, snr_target=snr_target,
    )


def write_mock_to_disk(mock: MockSpectrum, outdir: Path, prefix: str = "mock") -> dict:
    """Write .spec, template .dat, and Tlist files in the format kinextract.io
    expects. For continuum_mode="none" (oracle case), the caller should have
    already divided flux_noisy by true_continuum before building the mock
    (handled by scenarios.scenario_to_fitconfig, not here, since that's a
    per-continuum-mode fitting decision, not a mock-generation one).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    spec_path = outdir / f"{prefix}.spec"
    tmpl_path = outdir / f"{prefix}_tmpl.dat"
    tlist_path = outdir / "Tlist"
    write_spec_file(spec_path, mock.flux_noisy, mock.errors)
    write_template_file(tmpl_path, mock.template_wavelength, mock.template_flux)
    tlist_path.write_text(f"{tmpl_path.name}\n")
    return {"spec": spec_path, "template": tmpl_path, "tlist": tlist_path}
