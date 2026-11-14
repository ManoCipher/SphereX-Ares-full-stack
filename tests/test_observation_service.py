from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from spherex.observation_service import ObservationPreviewService


def test_service_resolves_downloaded_file_by_observation_id(tmp_path: Path):
    raw = tmp_path / "raw" / "target"
    raw.mkdir(parents=True)
    path = raw / "qr3_OBS_42_hash_cutout.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [6.0, 6.0]
    wcs.wcs.crval = [10.0, -2.0]
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    fits.ImageHDU(np.ones((12, 12)), header=wcs.to_header(), name="IMAGE").writeto(path)

    service = ObservationPreviewService(raw_dir=tmp_path / "raw", preview_dir=tmp_path / "previews")
    record = {"observation_id": "OBS_42", "detector": "2"}
    assert service.download_status(record) == "Downloaded"
    result = service.prepare_observation_previews([record], 10.0, -2.0, size=8)
    assert result[0]["status"] == "success"
    assert result[0]["fits_filename"] == path.name


def test_service_reports_unresolved_observation_without_stopping(tmp_path: Path):
    service = ObservationPreviewService(raw_dir=tmp_path / "raw", preview_dir=tmp_path / "previews")
    result = service.prepare_observation_previews([{"observation_id": "NOT_LOCAL"}], 10.0, -2.0)
    assert result[0]["status"] == "missing_file"
    assert "no downloaded FITS file path" in result[0]["error"]
