"""Tests for FitConfig and TOML loading."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kinextract import FitConfig, load_config_from_toml


def _write_toml(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    f.write(content)
    f.flush()
    return Path(f.name)


# ── FitConfig defaults ──────────────────────────────────────────────────────

def test_fitconfig_instantiates_with_defaults():
    cfg = FitConfig()
    assert cfg.icoff == 1
    assert cfg.xlam_auto is False        # default off; TOML config enables it
    assert cfg.fit_continuum is False # default off; TOML config enables it
    assert isinstance(cfg.xlam_auto_grid, (list, tuple))


def test_fitconfig_zgal_default_is_zero():
    cfg = FitConfig()
    assert cfg.zgal == 0.0


def test_fitconfig_cee_not_used_wrong():
    """Speed-of-light should be set correctly in code (not 2.99e5)."""
    from kinextract._utils import CEE
    assert abs(CEE - 299792.458) < 0.001, "CEE deviates from IAU value"


# ── TOML loading ────────────────────────────────────────────────────────────

MINIMAL_TOML = """
[wavelength]
zgal = 0.00157811

[kinematics]
sigl = 100.0
xlam_smooth_threshold = 0.25
"""


def test_load_config_from_toml_reads_zgal():
    p = _write_toml(MINIMAL_TOML)
    cfg = load_config_from_toml(p)
    assert abs(cfg.zgal - 0.00157811) < 1e-8


def test_load_config_from_toml_reads_threshold():
    p = _write_toml(MINIMAL_TOML)
    cfg = load_config_from_toml(p)
    assert abs(cfg.xlam_smooth_threshold - 0.25) < 1e-9


def test_load_config_from_toml_missing_file_raises():
    with pytest.raises(Exception):
        load_config_from_toml("/nonexistent/path/config.toml")


def test_load_config_from_toml_bad_section_uses_defaults():
    """Unknown keys should not crash; missing keys use FitConfig defaults."""
    p = _write_toml("[wavelength]\nzgal = 0.002\n")
    cfg = load_config_from_toml(p)
    assert abs(cfg.zgal - 0.002) < 1e-8
    assert cfg.sigl == FitConfig().sigl  # default preserved
