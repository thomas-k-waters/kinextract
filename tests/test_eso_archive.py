"""Tests for kinextract.eso_archive -- only the parts that don't
require network access. `query_muse_phase3` genuinely needs to reach the
ESO archive and isn't covered here (this package's own test suite must
not depend on external network availability); the `confirm=False` gate
in `download_dataset` is pure local logic and is fully testable offline.
"""
from __future__ import annotations

import pytest

from kinextract.eso_archive import download_dataset


def test_download_dataset_refuses_without_confirm():
    with pytest.raises(ValueError, match="confirm=True"):
        download_dataset("ADP.2017-03-22T15:25:58.757", "/tmp", confirm=False)


def test_download_dataset_refusal_message_includes_size_estimate_when_given():
    with pytest.raises(ValueError, match=r"~4\.5 GB"):
        download_dataset("ADP.2017-03-22T15:25:58.757", "/tmp", confirm=False, size_estimate_gb=4.5)
