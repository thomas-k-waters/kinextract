from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Iterable

import numpy as np
from numpy.polynomial import legendre
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None


SPEED_OF_LIGHT = 299792.458


@dataclass
class Spectrum:
    wave: np.ndarray
    flux: np.ndarray
    noise: np.ndarray | None = None


@dataclass
class FitConfig:
    name: str
    spectrograph: str
    wave_min: float | None = None
    wave_max: float | None = None
    redshift: float = 0.0
    deg_mpoly: int = 8
    deg_apoly: int = 0
    reg_lambda: float = 0.0
    n_mc: int = 100
    sigma_clip: float = 4.0
    max_clip_iter: int = 3
    n_velocity_bins: int = 29
    velocity_grid_from: Path | None = None
    fit_reference_grid: bool = True
    use_jax_backend: bool = False
    jax_enable_x64: bool = True
    random_seed: int | None = None
    rms: float | None = None
    bad_region_file: Path | None = None


@dataclass
class FitResult:
    name: str
    velocity: np.ndarray
    losvd: np.ndarray
    losvd_low: np.ndarray
    losvd_high: np.ndarray
    template_weights: np.ndarray
    model_wave: np.ndarray
    model_flux: np.ndarray
    galaxy_flux: np.ndarray
    noise: np.ndarray
    goodpixels: np.ndarray
    multiplicative_poly: np.ndarray
    additive_poly: np.ndarray
    log_wave: np.ndarray
    log_flux: np.ndarray
    log_noise: np.ndarray
    template_names: list[str]
    fit_rms: float
    fit_chi2: float
    fit_score: float
    summary_path: Path | None = None
    fit_path: Path | None = None
    rms_path: Path | None = None
    mcfit_path: Path | None = None
    mcfit2_path: Path | None = None
    sim_path: Path | None = None
    plot_path: Path | None = None


def read_ascii_spectrum(path: Path) -> Spectrum:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] == 2:
        wave = data[:, 0].astype(float)
        flux = data[:, 1].astype(float)
        noise = None
    elif data.shape[1] >= 3:
        wave = data[:, 0].astype(float)
        flux = data[:, 1].astype(float)
        noise = data[:, 2].astype(float)
    else:
        raise ValueError(f"Unsupported spectrum format in {path}")
    return Spectrum(wave=wave, flux=flux, noise=noise)


def read_flux_only_spectrum(path: Path, wave_min: float, wave_step: float) -> Spectrum:
    data = np.loadtxt(path)
    if data.ndim == 1:
        flux = np.asarray(data, dtype=float).reshape(-1)
        wave = wave_min + wave_step * np.arange(flux.size, dtype=float)
        return Spectrum(wave=wave, flux=flux, noise=None)

    if data.shape[1] >= 3:
        index = np.asarray(data[:, 0], dtype=float)
        flux = np.asarray(data[:, 1], dtype=float)
        noise = np.asarray(data[:, 2], dtype=float)
        start = float(np.nanmin(index)) if np.all(np.isfinite(index)) else 1.0
        wave = wave_min + wave_step * (index - start)
        return Spectrum(wave=wave, flux=flux, noise=noise)

    flux = np.asarray(data[:, 0], dtype=float)
    wave = wave_min + wave_step * np.arange(flux.size, dtype=float)
    noise = np.asarray(data[:, 1], dtype=float) if data.shape[1] >= 2 else None
    return Spectrum(wave=wave, flux=flux, noise=noise)


def read_template_list(path: Path) -> list[Path]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    templates: list[Path] = []
    for line in lines:
        if line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        templates.append(candidate)
    if not templates:
        raise ValueError(f"No templates listed in {path}")
    return templates


def load_templates(template_files: Iterable[Path]) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    names: list[str] = []
    waves: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    for file_path in template_files:
        table = np.loadtxt(file_path)
        if table.ndim == 1:
            table = table[None, :]
        if table.shape[1] < 2:
            raise ValueError(f"Template file {file_path} must have at least 2 columns")
        names.append(file_path.name)
        waves.append(table[:, 0].astype(float))
        fluxes.append(table[:, 1].astype(float))
    return names, waves, fluxes


def apply_redshift(wave: np.ndarray, redshift: float) -> np.ndarray:
    return wave / (1.0 + redshift)


def build_fit_mask(wave: np.ndarray, wave_min: float | None, wave_max: float | None) -> np.ndarray:
    mask = np.ones_like(wave, dtype=bool)
    if wave_min is not None:
        mask &= wave >= wave_min
    if wave_max is not None:
        mask &= wave <= wave_max
    return mask


def load_mask_intervals(path: Path | None) -> list[tuple[float, float]]:
    if path is None or not path.exists():
        return []
    intervals: list[tuple[float, float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        lo, hi = float(parts[0]), float(parts[1])
        intervals.append((min(lo, hi), max(lo, hi)))
    return intervals


def apply_mask_intervals(wave: np.ndarray, mask: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    if not intervals:
        return mask
    final = mask.copy()
    for lo, hi in intervals:
        final &= ~((wave >= lo) & (wave <= hi))
    return final


def flux_conserving_log_rebin(wave: np.ndarray, flux: np.ndarray, *, dlog: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    if wave.size < 4:
        raise ValueError("Spectrum is too short for log rebinning")
    if np.any(np.diff(wave) <= 0):
        order = np.argsort(wave)
        wave = wave[order]
        flux = flux[order]
    log_wave = np.log(wave)
    if dlog is None:
        dlog = float(np.median(np.diff(log_wave)))
    n_pix = wave.size
    log_edges = log_wave[0] + dlog * (np.arange(n_pix + 1) - 0.5)
    edges = np.exp(log_edges)
    cumulative = np.concatenate(([0.0], cumulative_trapezoid(flux, wave)))
    edge_cumulative = np.interp(edges, wave, cumulative, left=cumulative[0], right=cumulative[-1])
    rebinned = np.diff(edge_cumulative) / np.diff(edges)
    centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    velscale = SPEED_OF_LIGHT * dlog
    return centers, rebinned, velscale


def fit_flux_conserving_grid(
    spectrum: Spectrum,
    *,
    wave_min: float | None,
    wave_max: float | None,
    redshift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    wave = apply_redshift(spectrum.wave, redshift)
    mask = build_fit_mask(wave, wave_min, wave_max)
    wave = wave[mask]
    flux = spectrum.flux[mask]
    noise = None if spectrum.noise is None else spectrum.noise[mask]
    log_wave, log_flux, velscale = flux_conserving_log_rebin(wave, flux)
    if noise is None:
        log_noise = np.full_like(log_flux, np.std(log_flux) if np.std(log_flux) > 0 else 1.0)
    else:
        _, log_noise, _ = flux_conserving_log_rebin(wave, noise)
        positive = log_noise > 0
        log_noise = np.clip(log_noise, np.nanmedian(log_noise[positive]) if np.any(positive) else 1.0, np.inf)
    return log_wave, log_flux, log_noise, velscale


def rebin_template_to_grid(template_wave: np.ndarray, template_flux: np.ndarray, log_wave: np.ndarray) -> np.ndarray:
    if np.any(np.diff(template_wave) <= 0):
        order = np.argsort(template_wave)
        template_wave = template_wave[order]
        template_flux = template_flux[order]
    template_log_wave = np.log(template_wave)
    return np.interp(log_wave, template_log_wave, template_flux, left=0.0, right=0.0)


def shift_template(template_flux: np.ndarray, log_wave: np.ndarray, velocity: float) -> np.ndarray:
    delta = velocity / SPEED_OF_LIGHT
    shifted_log_wave = log_wave - delta
    return np.interp(shifted_log_wave, log_wave, template_flux, left=0.0, right=0.0)


def normalized_legendre_basis(x: np.ndarray, degree: int) -> np.ndarray:
    if degree < 0:
        raise ValueError("Polynomial degree must be non-negative")
    if degree == 0:
        return np.ones((x.size, 1), dtype=float)
    xx = np.clip(x, -1.0, 1.0)
    return legendre.legvander(xx, degree)


def center_and_scale_logwave(log_wave: np.ndarray) -> np.ndarray:
    if log_wave.size == 0:
        return log_wave
    lo, hi = float(log_wave[0]), float(log_wave[-1])
    if hi == lo:
        return np.zeros_like(log_wave)
    return 2.0 * (log_wave - lo) / (hi - lo) - 1.0


def build_template_design_matrix(
    template_grid: list[np.ndarray],
    log_wave: np.ndarray,
    velocity_grid: np.ndarray,
) -> np.ndarray:
    """Build (n_pix, n_vel * n_temp) design matrix via vectorized interpolation.

    Columns are ordered as [v0t0, v0t1, ..., v1t0, v1t1, ...] matching
    the original double-loop ordering. This fully-vectorized version avoids
    the Python-level loop over velocities.
    """
    n_pix = log_wave.size
    n_vel = velocity_grid.size
    n_temp = len(template_grid)
    templates = np.stack(template_grid, axis=0)           # (n_temp, n_pix)

    # Shifted log-wave query positions for every velocity: (n_vel, n_pix)
    shifted_lw = log_wave[None, :] - (velocity_grid[:, None] / SPEED_OF_LIGHT)

    # Linear interpolation indices and weights (clamp to valid range)
    span = log_wave[-1] - log_wave[0]
    frac = np.clip((shifted_lw - log_wave[0]) / span * (n_pix - 1), 0.0, n_pix - 1)
    idx_lo = np.clip(np.floor(frac).astype(int), 0, n_pix - 2)
    idx_hi = idx_lo + 1
    w_hi = (frac - idx_lo).astype(float)

    in_range = (shifted_lw >= log_wave[0]) & (shifted_lw <= log_wave[-1])

    # Gather interpolated values for all templates: (n_temp, n_vel, n_pix)
    vals = (1.0 - w_hi)[None] * templates[:, idx_lo] + w_hi[None] * templates[:, idx_hi]
    vals = np.where(in_range[None], vals, 0.0)

    # Reshape to (n_pix, n_vel * n_temp) with column ordering [vXtY]
    return vals.transpose(2, 1, 0).reshape(n_pix, n_vel * n_temp)


def build_template_design_matrix_jax(
    template_grid: list[np.ndarray],
    log_wave: np.ndarray,
    velocity_grid: np.ndarray,
) -> np.ndarray:
    """Build template design matrix with JAX on CPU; falls back if unavailable."""
    if jax is None or jnp is None:
        return build_template_design_matrix(template_grid, log_wave, velocity_grid)

    t = jnp.asarray(np.stack(template_grid, axis=0), dtype=jnp.float64)
    x = jnp.asarray(log_wave, dtype=jnp.float64)
    v = jnp.asarray(velocity_grid, dtype=jnp.float64)

    def _shift_one(temp_flux, vel):
        shifted_x = x - vel / SPEED_OF_LIGHT
        return jnp.interp(shifted_x, x, temp_flux, left=0.0, right=0.0)

    shifted = jax.vmap(
        lambda vel: jax.vmap(lambda temp_flux: _shift_one(temp_flux, vel))(t)
    )(v)
    # shifted shape: (n_vel, n_temp, n_pix) -> (n_pix, n_vel * n_temp)
    matrix = jnp.transpose(shifted, (2, 0, 1)).reshape((x.size, v.size * t.shape[0]))
    return np.asarray(matrix, dtype=float)


def build_losvd_regularization(n_vel: int, n_temp: int) -> np.ndarray:
    if n_vel < 3:
        return np.zeros((0, n_vel * n_temp), dtype=float)
    reg = np.zeros((n_vel - 2, n_vel * n_temp), dtype=float)
    for j in range(1, n_vel - 1):
        row = j - 1
        for k in range(n_temp):
            reg[row, (j - 1) * n_temp + k] = 1.0
            reg[row, j * n_temp + k] = -2.0
            reg[row, (j + 1) * n_temp + k] = 1.0
    return reg


def solve_weighted_bounded_lsq(
    design: np.ndarray,
    target: np.ndarray,
    noise: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    reg_matrix: np.ndarray | None = None,
    reg_strength: float = 0.0,
) -> np.ndarray:
    weighted_design = design / noise[:, None]
    weighted_target = target / noise
    if reg_matrix is not None and reg_matrix.size and reg_strength > 0:
        if reg_matrix.shape[1] < weighted_design.shape[1]:
            padded = np.zeros((reg_matrix.shape[0], weighted_design.shape[1]), dtype=float)
            padded[:, : reg_matrix.shape[1]] = reg_matrix
            reg_matrix = padded
        weighted_design = np.vstack([weighted_design, math.sqrt(reg_strength) * reg_matrix])
        weighted_target = np.concatenate([weighted_target, np.zeros(reg_matrix.shape[0], dtype=float)])
    result = lsq_linear(weighted_design, weighted_target, bounds=(lower, upper), method="trf", lsmr_tol="auto", max_iter=50)
    if result.success:
        return result.x

    solution, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    solution = np.asarray(solution, dtype=float)
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    solution[finite_lower] = np.maximum(solution[finite_lower], lower[finite_lower])
    solution[finite_upper] = np.minimum(solution[finite_upper], upper[finite_upper])
    return solution


def fit_with_polynomials(
    log_wave: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    template_grid: list[np.ndarray],
    velocity_grid: np.ndarray,
    *,
    deg_mpoly: int,
    deg_apoly: int,
    reg_lambda: float,
    sigma_clip: float,
    max_clip_iter: int,
    template_base: np.ndarray | None = None,
    use_jax_backend: bool = False,
) -> dict:
    x_scaled = center_and_scale_logwave(log_wave)
    additive_basis = normalized_legendre_basis(x_scaled, deg_apoly)
    ratio_basis = normalized_legendre_basis(x_scaled, deg_mpoly)
    mult_poly = np.ones_like(data)
    good = np.isfinite(data) & np.isfinite(noise) & (noise > 0) & np.isfinite(log_wave)
    n_temp = len(template_grid)
    n_vel = velocity_grid.size
    reg_matrix = build_losvd_regularization(n_vel, n_temp)
    if template_base is None:
        if use_jax_backend and jax is not None and jnp is not None:
            template_base = build_template_design_matrix_jax(template_grid, log_wave, velocity_grid)
        else:
            template_base = build_template_design_matrix(template_grid, log_wave, velocity_grid)

    template_coeffs = np.zeros((n_vel, n_temp), dtype=float)
    additive_coeffs = np.zeros(additive_basis.shape[1], dtype=float)
    poly_coeffs = np.array([1.0], dtype=float)
    final_model = np.zeros_like(data)
    final_template_model = np.zeros_like(data)

    for _ in range(max_clip_iter + 1):
        design_templates = template_base * mult_poly[:, None]
        design = np.hstack([design_templates, additive_basis])
        lower = np.zeros(design.shape[1], dtype=float)
        upper = np.full_like(lower, np.inf)
        if additive_basis.shape[1] > 0:
            lower[-additive_basis.shape[1]:] = -np.inf
        coeffs = solve_weighted_bounded_lsq(
            design[good],
            data[good],
            noise[good],
            lower,
            upper,
            reg_matrix=reg_matrix,
            reg_strength=reg_lambda,
        )
        n_template_coeffs = n_vel * n_temp
        template_flat = coeffs[:n_template_coeffs]
        additive_coeffs = coeffs[n_template_coeffs:]
        template_coeffs = template_flat.reshape(n_vel, n_temp)
        final_template_model = design_templates[:, :n_template_coeffs] @ template_flat
        if additive_basis.shape[1] > 0:
            final_model = final_template_model + additive_basis @ additive_coeffs
        else:
            final_model = final_template_model.copy()

        if ratio_basis.shape[1] > 1:
            ratio_target = np.clip(data / np.clip(final_model, 1e-12, np.inf), 0.0, 10.0)
            ratio_coeffs, *_ = np.linalg.lstsq(ratio_basis[good, 1:], ratio_target[good] - 1.0, rcond=None)
            poly_coeffs = np.concatenate([[1.0], ratio_coeffs])
            mult_poly_new = np.clip(ratio_basis @ poly_coeffs, 0.1, 10.0)
        else:
            mult_poly_new = np.ones_like(mult_poly)

        model = mult_poly_new * final_template_model
        if additive_basis.shape[1] > 0:
            model = model + additive_basis @ additive_coeffs

        residuals = (data - model) / noise
        clipped = good & (np.abs(residuals) <= sigma_clip)
        mult_poly = mult_poly_new
        final_model = model
        if np.array_equal(clipped, good):
            good = clipped
            break
        good = clipped

    template_weights = template_coeffs.sum(axis=0)
    losvd = template_coeffs.sum(axis=1)
    fit_rms = float(np.sqrt(np.mean(((data[good] - final_model[good]) / noise[good]) ** 2))) if np.any(good) else float("nan")
    fit_chi2 = float(np.sum(((data[good] - final_model[good]) / noise[good]) ** 2)) if np.any(good) else float("nan")
    return {
        "template_coeffs": template_coeffs,
        "template_weights": template_weights,
        "losvd": losvd,
        "model": final_model,
        "template_model": final_template_model,
        "multiplicative_poly": mult_poly,
        "additive_coeffs": additive_coeffs,
        "poly_coeffs": poly_coeffs,
        "goodpixels": good,
        "fit_rms": fit_rms,
        "fit_chi2": fit_chi2,
    }


def estimate_sigma_from_losvd(velocity_grid: np.ndarray, losvd: np.ndarray) -> float:
    if velocity_grid.size < 2 or np.sum(losvd) <= 0:
        return 60.0
    mean = float(np.sum(velocity_grid * losvd) / np.sum(losvd))
    var = float(np.sum(losvd * (velocity_grid - mean) ** 2) / np.sum(losvd))
    return max(20.0, math.sqrt(max(var, 1.0)))


def load_reference_grid(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 1:
        raise ValueError(f"Invalid reference grid in {path}")
    return data[:, 0].astype(float)


def choose_velocity_grid(
    *,
    velocity_grid_from: Path | None,
    estimated_sigma: float,
    n_velocity_bins: int,
) -> np.ndarray:
    if velocity_grid_from is not None and velocity_grid_from.exists():
        return load_reference_grid(velocity_grid_from)
    sigma = max(float(estimated_sigma), 20.0)
    half_range = 4.5 * sigma
    return np.linspace(-half_range, half_range, n_velocity_bins)


def write_fit_summary(path: Path, log_wave: np.ndarray, data: np.ndarray, model: np.ndarray, goodpixels: np.ndarray) -> None:
    mask_int = goodpixels.astype(int)
    out = np.column_stack([np.exp(log_wave), data, model, data - model, mask_int])
    np.savetxt(path, out, fmt=["%.8f", "%.8f", "%.8f", "%.8f", "%d"])


def write_rms_file(path: Path, fit_rms: float, n_good: int, reg_lambda: float) -> None:
    path.write_text(f"{fit_rms:.10f} {n_good:d} {reg_lambda:.10f}\n")


def write_mcfit_files(
    stem: Path,
    velocity_grid: np.ndarray,
    losvd: np.ndarray,
    losvd_low: np.ndarray,
    losvd_high: np.ndarray,
    template_weights: np.ndarray,
    template_weight_samples: np.ndarray | None = None,
    losvd_samples: np.ndarray | None = None,
) -> tuple[Path, Path, Path]:
    mcfit_path = stem.with_suffix(".mcfit")
    mcfit2_path = stem.with_suffix(".mcfit2")
    sim_path = stem.with_suffix(".sim")

    n_vel = velocity_grid.size
    n_temp = template_weights.size
    mcfit_rows = np.column_stack([
        velocity_grid,
        losvd,
        losvd_low,
        losvd_high,
        np.zeros_like(velocity_grid),
        np.zeros_like(velocity_grid),
    ])
    with mcfit_path.open("w") as handle:
        handle.write(f"{n_vel:d} {n_temp:d}\n")
        np.savetxt(handle, mcfit_rows, fmt="%.8f %.8f %.8f %.8f %.8f %.8f")

    with mcfit2_path.open("w") as handle:
        np.savetxt(handle, np.column_stack([velocity_grid, losvd, losvd_low, losvd_high]), fmt="%.8f %.8f %.8f %.8f")

    if template_weight_samples is None:
        template_weight_samples = np.zeros((1, n_temp), dtype=float)
    nsim = template_weight_samples.shape[0]
    # losvd_samples shape: (nsim, n_vel); fall back to repeating MAP losvd when absent.
    if losvd_samples is None or losvd_samples.shape[0] != nsim:
        losvd_cols = np.tile(losvd[:, None], (1, nsim))  # (n_vel, nsim)
    else:
        losvd_cols = losvd_samples.T  # (n_vel, nsim)
    with sim_path.open("w") as handle:
        handle.write(f"{nsim:d} {n_vel:d} {n_temp:d}\n")
        handle.write(" ".join(f"{x:.8f}" for x in np.linspace(0.0, 1.0, nsim)) + "\n")
        for j in range(n_vel):
            handle.write(" ".join(f"{v:.8f}" for v in losvd_cols[j]) + "\n")
        for k in range(n_temp):
            handle.write(" ".join(f"{template_weights[k]:.8f}" for _ in range(nsim)) + "\n")
        handle.write(" ".join("0.0" for _ in range(nsim)) + "\n")
    return mcfit_path, mcfit2_path, sim_path


def fit_spectrum(
    spectrum_path: Path,
    template_files: list[Path],
    config: FitConfig,
    *,
    spectrum_kind: str = "two-column",
    flux_only_wave_min: float | None = None,
    flux_only_step: float | None = None,
) -> FitResult:
    progress_prefix = f"[{config.name}]"
    if spectrum_kind == "flux-only" or (flux_only_wave_min is not None and flux_only_step is not None):
        if flux_only_wave_min is None or flux_only_step is None:
            raise ValueError("Flux-only spectra require wave_min and wave_step")
        spectrum = read_flux_only_spectrum(spectrum_path, flux_only_wave_min, flux_only_step)
    else:
        spectrum = read_ascii_spectrum(spectrum_path)

    spectrum = Spectrum(
        wave=apply_redshift(spectrum.wave, config.redshift),
        flux=spectrum.flux.astype(float),
        noise=spectrum.noise.astype(float) if spectrum.noise is not None else None,
    )
    base_mask = build_fit_mask(spectrum.wave, config.wave_min, config.wave_max)
    intervals = load_mask_intervals(config.bad_region_file)
    base_mask = apply_mask_intervals(spectrum.wave, base_mask, intervals)
    spectrum = Spectrum(
        wave=spectrum.wave[base_mask],
        flux=spectrum.flux[base_mask],
        noise=None if spectrum.noise is None else spectrum.noise[base_mask],
    )

    log_wave, log_flux, log_noise, _ = fit_flux_conserving_grid(
        spectrum,
        wave_min=None,
        wave_max=None,
        redshift=0.0,
    )

    template_names, template_waves, template_fluxes = load_templates(template_files)
    if len(template_waves) != len(template_fluxes):
        raise ValueError("Template wavelength/flux arrays are mismatched")
    template_grid = [rebin_template_to_grid(w, f, log_wave) for w, f in zip(template_waves, template_fluxes)]

    if config.rms is not None:
        log_noise = np.full_like(log_flux, float(config.rms))

    if config.use_jax_backend and jax is not None and config.jax_enable_x64:
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

    sigma_guess = 60.0
    print(f"{progress_prefix} fitting primary spectrum", flush=True)
    if config.velocity_grid_from is not None and config.velocity_grid_from.exists():
        velocity_grid = load_reference_grid(config.velocity_grid_from)
    else:
        velocity_grid = choose_velocity_grid(
            velocity_grid_from=None,
            estimated_sigma=sigma_guess,
            n_velocity_bins=config.n_velocity_bins,
        )

    def _build_template_base(current_velocity_grid: np.ndarray) -> np.ndarray:
        if config.use_jax_backend and jax is not None and jnp is not None:
            return build_template_design_matrix_jax(template_grid, log_wave, current_velocity_grid)
        return build_template_design_matrix(template_grid, log_wave, current_velocity_grid)

    template_base = _build_template_base(velocity_grid)

    fit = fit_with_polynomials(
        log_wave,
        log_flux,
        log_noise,
        template_grid,
        velocity_grid,
        deg_mpoly=config.deg_mpoly,
        deg_apoly=config.deg_apoly,
        reg_lambda=config.reg_lambda,
        sigma_clip=config.sigma_clip,
        max_clip_iter=config.max_clip_iter,
        template_base=template_base,
        use_jax_backend=config.use_jax_backend,
    )

    if config.fit_reference_grid and config.velocity_grid_from is None:
        estimated_sigma = estimate_sigma_from_losvd(velocity_grid, fit["losvd"])
        print(f"{progress_prefix} refining velocity grid", flush=True)
        velocity_grid = choose_velocity_grid(
            velocity_grid_from=None,
            estimated_sigma=estimated_sigma,
            n_velocity_bins=config.n_velocity_bins,
        )
        template_base = _build_template_base(velocity_grid)
        fit = fit_with_polynomials(
            log_wave,
            log_flux,
            log_noise,
            template_grid,
            velocity_grid,
            deg_mpoly=config.deg_mpoly,
            deg_apoly=config.deg_apoly,
            reg_lambda=config.reg_lambda,
            sigma_clip=config.sigma_clip,
            max_clip_iter=config.max_clip_iter,
            template_base=template_base,
            use_jax_backend=config.use_jax_backend,
        )

    losvd = np.clip(fit["losvd"], 0.0, np.inf)
    if np.sum(losvd) > 0:
        losvd = losvd / np.sum(losvd)
    # Heuristic ±12% envelope used only when n_mc == 0; replaced by MC percentiles below.
    low = np.maximum(0.0, losvd - np.maximum(0.005, 0.12 * losvd))
    high = np.maximum(losvd + np.maximum(0.005, 0.12 * losvd), losvd)
    template_weights = np.clip(fit["template_weights"], 0.0, np.inf)
    if np.sum(template_weights) > 0:
        template_weights = template_weights / np.sum(template_weights)

    rng = np.random.default_rng(config.random_seed)
    template_weight_samples: list[np.ndarray] = []
    losvd_samples: list[np.ndarray] = []
    fit_scores: list[float] = []

    if config.n_mc > 0:
        for _ in range(config.n_mc):
            print(f"{progress_prefix} MC refit {_ + 1}/{config.n_mc}", flush=True)
            perturbed = fit["model"] + rng.normal(0.0, log_noise)
            noisy_fit = fit_with_polynomials(
                log_wave,
                perturbed,
                log_noise,
                template_grid,
                velocity_grid,
                deg_mpoly=config.deg_mpoly,
                deg_apoly=config.deg_apoly,
                reg_lambda=config.reg_lambda,
                sigma_clip=config.sigma_clip,
                max_clip_iter=config.max_clip_iter,
                template_base=template_base,
                use_jax_backend=config.use_jax_backend,
            )
            sample_losvd = np.clip(noisy_fit["losvd"], 0.0, np.inf)
            if np.sum(sample_losvd) > 0:
                sample_losvd = sample_losvd / np.sum(sample_losvd)
            sample_weights = np.clip(noisy_fit["template_weights"], 0.0, np.inf)
            if np.sum(sample_weights) > 0:
                sample_weights = sample_weights / np.sum(sample_weights)
            losvd_samples.append(sample_losvd)
            template_weight_samples.append(sample_weights)
            fit_scores.append(float(noisy_fit["fit_rms"]))

        losvd_stack = np.vstack(losvd_samples)
        low = np.percentile(losvd_stack, 16, axis=0)
        losvd = np.percentile(losvd_stack, 50, axis=0)
        high = np.percentile(losvd_stack, 84, axis=0)
        losvd = np.clip(losvd, 0.0, np.inf)
        low = np.clip(np.minimum(low, losvd), 0.0, np.inf)
        high = np.maximum(high, losvd)
        template_weight_samples_arr = np.vstack(template_weight_samples)
    else:
        template_weight_samples_arr = np.array([template_weights], dtype=float)

    summary = FitResult(
        name=config.name,
        velocity=velocity_grid,
        losvd=losvd,
        losvd_low=low,
        losvd_high=high,
        template_weights=template_weights,
        model_wave=np.exp(log_wave),
        model_flux=fit["model"],
        galaxy_flux=log_flux,
        noise=log_noise,
        goodpixels=fit["goodpixels"],
        multiplicative_poly=fit["multiplicative_poly"],
        additive_poly=fit["additive_coeffs"],
        log_wave=log_wave,
        log_flux=log_flux,
        log_noise=log_noise,
        template_names=template_names,
        fit_rms=float(fit["fit_rms"]),
        fit_chi2=float(fit["fit_chi2"]),
        fit_score=float(np.median(fit_scores)) if fit_scores else float(fit["fit_rms"]),
    )

    summary._template_weight_samples = template_weight_samples_arr  # type: ignore[attr-defined]
    if config.n_mc > 0:
        summary._losvd_samples = losvd_stack  # type: ignore[attr-defined]
    return summary


def save_fit_products(
    result: FitResult,
    output_dir: Path,
    *,
    write_plot: bool = True,
    write_mcfit_products: bool = True,
) -> FitResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / result.name
    fit_path = stem.with_suffix(".fit")
    rms_path = stem.with_suffix(".rms")
    summary_path = stem.with_suffix(".json")
    plot_path = stem.with_name(stem.name + "_specfit.pdf")

    write_fit_summary(fit_path, result.log_wave, result.galaxy_flux, result.model_flux, result.goodpixels)
    write_rms_file(rms_path, result.fit_rms, int(np.sum(result.goodpixels)), 0.0)
    if write_mcfit_products:
        template_weight_samples = getattr(result, "_template_weight_samples", None)
        losvd_samples = getattr(result, "_losvd_samples", None)
        mcfit_path, mcfit2_path, sim_path = write_mcfit_files(
            stem,
            result.velocity,
            result.losvd,
            result.losvd_low,
            result.losvd_high,
            result.template_weights,
            template_weight_samples=template_weight_samples,
            losvd_samples=losvd_samples,
        )
    else:
        mcfit_path = None
        mcfit2_path = None
        sim_path = None

    payload = {
        "name": result.name,
        "velocity": result.velocity.tolist(),
        "losvd": result.losvd.tolist(),
        "losvd_low": result.losvd_low.tolist(),
        "losvd_high": result.losvd_high.tolist(),
        "template_weights": result.template_weights.tolist(),
        "fit_rms": result.fit_rms,
        "fit_chi2": result.fit_chi2,
        "fit_score": result.fit_score,
        "template_names": result.template_names,
    }
    summary_path.write_text(json.dumps(payload, indent=2))

    result.summary_path = summary_path
    result.fit_path = fit_path
    result.rms_path = rms_path
    result.mcfit_path = mcfit_path
    result.mcfit2_path = mcfit2_path
    result.sim_path = sim_path
    result.plot_path = plot_path

    if write_plot:
        plot_spectrum_fit(result, plot_path)

    return result


def plot_spectrum_fit(result: FitResult, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    wave = result.model_wave
    data = result.galaxy_flux
    model = result.model_flux
    residual = data - model
    good = result.goodpixels
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax0.plot(wave, data, color="black", lw=1.0, label="data")
    ax0.plot(wave, model, color="#bf5b17", lw=1.2, label="model")
    if not np.all(good):
        ax0.scatter(wave[~good], data[~good], s=8, color="#7f7f7f", alpha=0.5, label="masked")
    ax0.set_ylabel("Flux")
    ax0.legend(loc="best", frameon=False)
    ax0.set_title(result.name)
    ax1.axhline(0.0, color="#444444", lw=0.8)
    ax1.plot(wave, residual, color="#2c7fb8", lw=1.0)
    ax1.set_ylabel("Residual")
    ax1.set_xlabel("Wavelength")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_losvd(result: FitResult, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(result.velocity, result.losvd_low, result.losvd_high, color="#c7e9b4", alpha=0.7, label="1σ envelope")
    ax.plot(result.velocity, result.losvd, color="#006d2c", lw=1.8, label="LOSVD")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Relative light")
    ax.legend(frameon=False)
    ax.set_title(result.name)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
