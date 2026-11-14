from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from spherex.fits_processor import (
    InvalidImageDataError,
    InvalidWcsError,
    SpherexFitsProcessor,
    TargetOutsideImageError,
    inspect_directory,
)


def _wcs_header(*, ra: float = 150.25, dec: float = 2.1) -> fits.Header:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [51.0, 51.0]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    header = wcs.to_header()
    header["BUNIT"] = "MJy/sr"
    header["OBSID"] = "TEST_001"
    header["DETECTOR"] = "3"
    header["DATE-OBS"] = "2026-04-28T12:00:00"
    return header


def _write_image(path: Path, data: np.ndarray | None = None, *, wcs: bool = True) -> None:
    image = np.arange(10000, dtype=np.float32).reshape(100, 100) if data is None else data
    header = _wcs_header() if wcs else fits.Header({"BUNIT": "MJy/sr"})
    fits.HDUList(
        [
            fits.PrimaryHDU(np.zeros((10, 10), dtype=np.float32)),
            fits.ImageHDU(image, header=header, name="IMAGE"),
            fits.ImageHDU(np.ones_like(image), name="VARIANCE"),
        ]
    ).writeto(path)


def test_synthetic_mef_inventory_and_plot(tmp_path: Path):
    raw = tmp_path / "raw"
    plots = tmp_path / "plots"
    raw.mkdir()
    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((8, 8)), name="IMAGE"),
        fits.ImageHDU(np.zeros((8, 8), dtype=np.uint16), name="FLAGS"),
        fits.ImageHDU(np.ones((8, 8)), name="VARIANCE"),
        fits.ImageHDU(np.ones((8, 8)), name="ZODI"),
        fits.ImageHDU(np.ones((2, 4, 4)), name="PSF"),
        fits.BinTableHDU.from_columns([], name="WCS-WAVE"),
    ]
    fits.HDUList(hdus).writeto(raw / "synthetic.fits")
    report = inspect_directory(raw, plots)
    classes = set(report.classification)
    assert "science_candidate" in classes
    assert "variance_or_uncertainty" in classes
    assert "mask_or_quality" in classes
    assert "psf" in classes
    assert "background_model" in classes
    assert "wavelength_calibration" in classes
    assert list(plots.glob("*.png"))


def test_inspection_prefers_named_image_hdu_and_extracts_metadata(tmp_path: Path):
    path = tmp_path / "science.fits"
    _write_image(path)
    metadata = SpherexFitsProcessor(tmp_path / "previews").inspect_fits(path)
    assert metadata["science_hdu_index"] == 1
    assert metadata["image_width"] == 100
    assert metadata["dtype"] == ">f4"
    assert metadata["observation_id"] == "TEST_001"
    assert metadata["detector"] == "3"
    assert metadata["brightness_unit"] == "MJy/sr"
    assert metadata["wcs"]["has_celestial"] is True


def test_wcs_conversion_and_cutout_alignment(tmp_path: Path):
    path = tmp_path / "science.fits"
    _write_image(path)
    processor = SpherexFitsProcessor(tmp_path / "previews")
    processor.configure_cleaning({"enabled": False})
    result = processor.extract_target(path, 150.25, 2.1, size=20)
    assert result["metadata"]["pixel_x"] == pytest.approx(50.0, abs=1e-6)
    assert result["metadata"]["pixel_y"] == pytest.approx(50.0, abs=1e-6)
    assert result["data"].shape == (20, 20)
    cutout_x, cutout_y = result["wcs"].world_to_pixel(SkyCoord(150.25, 2.1, unit="deg"))
    assert cutout_x == pytest.approx(10.0, abs=1e-6)
    assert cutout_y == pytest.approx(10.0, abs=1e-6)
    assert result["data"][10, 10] == pytest.approx(5050.0)
    assert np.array_equal(result["data"], result["original_data"])


def test_target_outside_and_missing_wcs_are_distinct(tmp_path: Path):
    processor = SpherexFitsProcessor(tmp_path / "previews")
    valid = tmp_path / "valid.fits"
    missing = tmp_path / "missing_wcs.fits"
    _write_image(valid)
    _write_image(missing, wcs=False)
    with pytest.raises(TargetOutsideImageError, match="outside image footprint"):
        processor.extract_target(valid, 151.0, 2.1, size=20)
    with pytest.raises(InvalidWcsError, match="no celestial WCS"):
        processor.extract_target(missing, 150.25, 2.1, size=20)


def test_nan_handling_and_png_generation(tmp_path: Path):
    data = np.full((100, 100), np.nan, dtype=np.float32)
    data[45:56, 45:56] = 7.0
    path = tmp_path / "nan_image.fits"
    _write_image(path, data)
    processor = SpherexFitsProcessor(tmp_path / "previews")
    cutout = processor.extract_target(path, 150.25, 2.1, size=20)
    rendered = processor.render_preview(cutout, tmp_path / "preview.png")
    assert Path(rendered["preview_path"]).is_file()
    assert rendered["finite_pixel_count"] == 121
    assert rendered["constant_image"] is True

    empty = tmp_path / "empty.fits"
    _write_image(empty, np.full((100, 100), np.nan, dtype=np.float32))
    with pytest.raises(InvalidImageDataError, match="no finite"):
        processor.extract_target(empty, 150.25, 2.1, size=20)


def test_preview_cache_and_multiple_observation_partial_failure(tmp_path: Path):
    path = tmp_path / "TEST_001.fits"
    _write_image(path)
    processor = SpherexFitsProcessor(tmp_path / "previews")
    good = {
        "observation_id": "TEST_001",
        "observation_timestamp": "2026-04-28T12:00:00",
        "detector": "3",
        "file_path": str(path),
    }
    missing = {"observation_id": "MISSING", "file_path": str(tmp_path / "missing.fits")}
    first = processor.prepare_observation_previews([good, missing], 150.25, 2.1, size=20)
    second = processor.prepare_observation_previews([good], 150.25, 2.1, size=20)
    assert first[0]["status"] == "success"
    assert first[0]["cached"] is False
    assert first[1]["status"] == "missing_file"
    assert second[0]["status"] == "success"
    assert second[0]["cached"] is True
    assert second[0]["pixel_x"] == pytest.approx(50.0)
    assert second[0]["preview_path"] == first[0]["preview_path"]


def test_selection_limit_is_enforced(tmp_path: Path):
    processor = SpherexFitsProcessor(tmp_path / "previews")
    with pytest.raises(ValueError, match="no more than 6"):
        processor.prepare_observation_previews([{}] * 7, 150.25, 2.1)


def test_rendering_settings_change_preview_cache_key(tmp_path: Path):
    path = tmp_path / "science.fits"
    _write_image(path)
    processor = SpherexFitsProcessor(tmp_path / "previews")
    gray = processor.prepare_observation_previews(
        [{"observation_id": "TEST", "file_path": str(path)}],
        150.25,
        2.1,
        size=20,
        rendering={"mode": "greyscale", "colormap": "gray", "stretch": "linear"},
    )[0]
    inferno = processor.prepare_observation_previews(
        [{"observation_id": "TEST", "file_path": str(path)}],
        150.25,
        2.1,
        size=20,
        rendering={"mode": "false_colour", "colormap": "inferno", "stretch": "asinh"},
    )[0]
    assert gray["preview_path"] != inferno["preview_path"]
    assert gray["rendering"]["colormap"] == "gray"
    assert inferno["rendering"]["colormap"] == "inferno"


def test_wcs_wave_image_is_sampled_at_target_pixel(tmp_path: Path):
    path = tmp_path / "wavelength.fits"
    wavelength = np.full((100, 100), 2.75, dtype=np.float32)
    wavelength_header = fits.Header({"BUNIT": "um"})
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((100, 100)), header=_wcs_header(), name="IMAGE"),
            fits.ImageHDU(wavelength, header=wavelength_header, name="WCS-WAVE"),
        ]
    ).writeto(path)
    result = SpherexFitsProcessor(tmp_path / "previews").wavelength_at_target(path, 50.0, 50.0)
    assert result["wavelength_um"] == pytest.approx(2.75)
    assert result["source"] == "FITS WCS-WAVE image at target"
