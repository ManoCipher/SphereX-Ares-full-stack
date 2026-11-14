from pathlib import Path

import numpy as np
import pandas as pd
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


def test_service_does_not_reuse_same_id_file_from_another_detector(tmp_path: Path):
    raw = tmp_path / "raw" / "viewer"
    raw.mkdir(parents=True)
    for detector in (3, 6):
        path = raw / f"release_SHARED_hash{detector}_cutout.fits"
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crpix = [6.0, 6.0]
        wcs.wcs.crval = [10.0, -2.0]
        wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
        header = wcs.to_header()
        header["OBSID"] = "SHARED"
        header["DETECTOR"] = detector
        fits.ImageHDU(np.full((12, 12), detector), header=header, name="IMAGE").writeto(path)

    service = ObservationPreviewService(raw_dir=tmp_path / "raw", preview_dir=tmp_path / "previews")
    path3 = service.resolve_fits_path({"observation_id": "SHARED", "detector": "3"})
    path6 = service.resolve_fits_path({"observation_id": "SHARED", "detector": "6"})
    assert path3 is not None and "hash3" in path3.name
    assert path6 is not None and "hash6" in path6.name
    assert path3 != path6


def test_service_rejects_duplicate_product_selection(tmp_path: Path):
    service = ObservationPreviewService(raw_dir=tmp_path / "raw", preview_dir=tmp_path / "previews")
    record = {"observation_id": "SAME", "detector": "2", "fits_url": "https://example.test/a.fits"}
    try:
        service.prepare_observation_previews([record, record], 10.0, -2.0)
    except ValueError as exc:
        assert "same observation product" in str(exc)
    else:
        raise AssertionError("duplicate product selection was not rejected")


def test_service_reports_unresolved_observation_without_stopping(tmp_path: Path):
    service = ObservationPreviewService(raw_dir=tmp_path / "raw", preview_dir=tmp_path / "previews")
    result = service.prepare_observation_previews([{"observation_id": "NOT_LOCAL"}], 10.0, -2.0)
    assert result[0]["status"] == "missing_file"
    assert "no FITS download URL" in result[0]["error"]


def test_service_downloads_missing_selected_file_then_previews(tmp_path: Path):
    class FakeDownloader:
        def __init__(self) -> None:
            self.requested_ids: list[str] = []

        def download(self, observations, output_dir, *, max_files=None):
            output_dir.mkdir(parents=True, exist_ok=True)
            row = observations.iloc[0]
            self.requested_ids = list(observations["obs_id"])
            filename = f"qr3_{row['obs_id']}_cutout.fits"
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcs.wcs.crpix = [101.0, 101.0]
            wcs.wcs.crval = [float(row["target_ra_deg"]), float(row["target_dec_deg"])]
            wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
            fits.ImageHDU(
                np.ones((200, 200)), header=wcs.to_header(), name="IMAGE"
            ).writeto(output_dir / filename)
            return pd.DataFrame(
                [{"obs_id": row["obs_id"], "filename": filename, "status": "downloaded", "error": None}]
            )

    downloader = FakeDownloader()
    service = ObservationPreviewService(
        raw_dir=tmp_path / "raw",
        preview_dir=tmp_path / "previews",
        downloader=downloader,
    )
    result = service.prepare_observation_previews(
        [
            {
                "observation_id": "OBS_NEW",
                "fits_url": "https://irsa.ipac.caltech.edu/ibe/data/spherex/example.fits",
                "collection": "spherex_qr3",
                "detector": "4",
            }
        ],
        10.0,
        -2.0,
        size=20,
    )
    assert downloader.requested_ids == ["OBS_NEW"]
    assert result[0]["status"] == "success"
    assert result[0]["fits_filename"] == "qr3_OBS_NEW_cutout.fits"
    assert Path(result[0]["preview_path"]).is_file()
