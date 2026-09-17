"""Single source of truth for the package version.

Kept as its own tiny, dependency-free module (not folded into
`__init__.py`) so both `pyproject.toml` (via
`[tool.setuptools.dynamic]`) and `.github/workflows/publish.yml`'s own
release-tag/package-version consistency check can read it without
importing the full package (and its numpy/jax/astropy dependencies).
"""
__version__ = "1.0.1"
