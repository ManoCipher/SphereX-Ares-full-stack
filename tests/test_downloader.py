from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from spherex.config import ArchiveConfig
from spherex.downloader import MAX_FILES_HARD_LIMIT, CutoutDownloader, build_cutout_url, validate_fits


BASE_URL = "https://irsa.ipac.caltech.edu/ibe/data/spherex/qr3/level2/p/v/1/example.fits"


def observation(url: str = BASE_URL) -> pd.DataFrame:
    return pd.DataFrame([{
        "obs_id": "obs/1", "data_release": "qr3", "access_url": url,
        "target_ra_deg": 133.7, "target_dec_deg": -7.2,
    }])


def test_cutout_url_and_full_image_guard():
    result = build_cutout_url(BASE_URL, 1.2, -3.4, 0.02)
    assert "center=1.20000000%2C-3.40000000" in result
    assert "size=0.020000" in result
    with pytest.raises(ValueError, match="Refusing"):
        build_cutout_url("https://example.org/whole.fits", 1, 2, 0.02)


def test_duplicate_handling_in_dry_run(tmp_path: Path):
    rows = pd.concat([observation(), observation()], ignore_index=True)
    manifest = CutoutDownloader().download(rows, tmp_path, dry_run=True)
    assert list(manifest.status) == ["dry_run", "duplicate"]


def test_explicit_twenty_file_dry_run(tmp_path: Path):
    rows = pd.concat(
        [observation(BASE_URL.replace("example.fits", f"example-{index}.fits")) for index in range(25)],
        ignore_index=True,
    )
    manifest = CutoutDownloader().download(rows, tmp_path, max_files=20, dry_run=True)
    assert len(manifest) == 20
    assert MAX_FILES_HARD_LIMIT >= 20


class FakeResponse:
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        yield from self.chunks


class FakeSession:
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    def mount(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return FakeResponse(self.chunks)


def test_budget_enforcement_removes_partial(tmp_path: Path):
    downloader = CutoutDownloader(ArchiveConfig(max_mb=0.000001), session=FakeSession([b"too large"]))
    manifest = downloader.download(observation(), tmp_path, max_mb=0.000001)
    assert manifest.loc[0, "status"] == "rejected"
    assert "budget" in manifest.loc[0, "error"]
    assert not list(tmp_path.glob("*.part"))


def test_fits_validation(tmp_path: Path):
    valid = tmp_path / "valid.fits"
    fits.PrimaryHDU(np.zeros((3, 3), dtype=np.float32)).writeto(valid)
    validate_fits(valid)
    invalid = tmp_path / "invalid.fits"
    invalid.write_bytes(b"not fits")
    with pytest.raises(Exception):
        validate_fits(invalid)
