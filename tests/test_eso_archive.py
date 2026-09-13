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


def test_download_dataset_reuses_existing_local_file_without_confirm(tmp_path):
    dp_id = "ADP.2017-03-22T15:25:58.757"
    existing = tmp_path / f"{dp_id}.fits"
    existing.write_bytes(b"")  # empty stand-in -- only the path/glob match matters here

    # confirm defaults to False -- if this reached the network-gate check at all, it
    # would raise ValueError like the tests above. Returning the existing path instead
    # is the whole point of the fix: an already-downloaded dataset should never need
    # confirm=True (or a network call) again.
    files = download_dataset(dp_id, str(tmp_path))
    assert files == [str(existing)]


def test_download_dataset_force_bypasses_existing_local_file_and_still_refuses_without_confirm(tmp_path):
    dp_id = "ADP.2017-03-22T15:25:58.757"
    (tmp_path / f"{dp_id}.fits").write_bytes(b"")

    with pytest.raises(ValueError, match="confirm=True"):
        download_dataset(dp_id, str(tmp_path), confirm=False, force=True)
