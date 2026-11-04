from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from spherex.scientific_visualization import (
    analyze_fits,
    aperture_photometry,
    pixel_is_inside,
    target_pixel,
)
from spherex.targets import WISE_0855


def _image_header(time: Time) -> fits.Header:
    coord = WISE_0855.at_observer(time)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [6.0, 6.0]
    wcs.wcs.crval = [coord.ra.deg, coord.dec.deg]
    wcs.wcs.cdelt = [-6.15 / 3600, 6.15 / 3600]
    header = wcs.to_header()
    header["EXTNAME"] = "IMAGE"
    header["MJD-AVG"] = time.mjd
    header["TIMESYS"] = "UTC"
    header["OBSID"] = "TEST_OBS"
    header["DETECTOR"] = 1
    header["BUNIT"] = "MJy / sr"
    return header


def test_spatial_wcs_maps_propagated_target_to_reference_pixel():
    time = Time("2026-04-23T10:35:00", scale="utc")
    header = _image_header(time)
    _, x, y, inside = target_pixel(header, (12, 12), WISE_0855, time)
    assert inside
    assert x == pytest.approx(5.0, abs=1e-5)
    assert y == pytest.approx(5.0, abs=1e-5)


def test_cutout_boundaries_use_pixel_edges():
    shape = (12, 12)
    assert pixel_is_inside(-0.5, 11.5, shape)
    assert not pixel_is_inside(-0.5001, 5.0, shape)
    assert not pixel_is_inside(5.0, 11.5001, shape)


def test_invalid_variance_cannot_produce_snr():
    image = np.ones((12, 12))
    variance = np.zeros((12, 12))
    result = aperture_photometry(image, variance, np.zeros_like(image, dtype=bool), 5.0, 5.0)
    assert result["status"] == "INCONCLUSIVE"
    assert np.isnan(result["aperture_snr"])


def test_missing_extensions_are_reported_not_crashed(tmp_path: Path):
    time = Time("2026-04-23T10:35:00", scale="utc")
    path = tmp_path / "qr2_missing.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(np.ones((12, 12)), header=_image_header(time), name="IMAGE")]
    ).writeto(path)
    row = analyze_fits(path, tmp_path / "plots", WISE_0855, make_plot=False)
    assert row["detection_status"] == "INCONCLUSIVE"
    assert "missing extension" in row["warnings"]
