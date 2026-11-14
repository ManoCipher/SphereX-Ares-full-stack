from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from spherex.fits_processor import SpherexFitsProcessor
from spherex.image_cleaner import SpherexImageCleaner


def _header() -> fits.Header:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [16.0, 16.0]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    return wcs.to_header()


def test_cleaner_keeps_masks_separate_and_preserves_sources():
    y, x = np.mgrid[:32, :32]
    background = 10.0 + 0.02 * x + 0.01 * y
    image = background.copy()
    image[16, 16] += 100.0
    flags = np.zeros(image.shape, dtype=np.uint32)
    flags[2, 2] |= 1 << 6  # NONFUNC: invalid science
    flags[3, 3] |= 1 << 21  # SOURCE: exclude from background, preserve measurement
    flags[4, 4] |= 1 << 25  # STREAK: diagnostic only, do not erase
    variance = np.ones(image.shape)
    original = image.copy()

    result = SpherexImageCleaner().clean(
        image, flags=flags, variance=variance, header=_header()
    )

    assert np.array_equal(image, original)
    assert result["original_image"] is image
    assert result["quality_mask"][2, 2]
    assert np.isnan(result["cleaned_image"][2, 2])
    assert not result["quality_mask"][3, 3]
    assert np.isfinite(result["cleaned_image"][3, 3])
    assert result["artefact_mask"][4, 4]
    assert np.isfinite(result["cleaned_image"][4, 4])
    assert result["cleaned_image"][16, 16] > 90
    assert result["original_wcs"].has_celestial


def test_zodi_is_subtracted_once_and_can_be_disabled():
    image = np.full((20, 20), 12.0)
    zodi = np.full_like(image, 2.0)
    cleaner = SpherexImageCleaner()
    cleaned = cleaner.clean(image, zodi=zodi)
    disabled = cleaner.clean(image, zodi=zodi, settings={"enabled": False})
    marked = cleaner.clean(image, zodi=zodi, header={"ZODISUB": True})

    assert cleaned["provenance"]["zodi_action"] == "subtracted_once"
    assert disabled["provenance"]["zodi_action"] == "not_subtracted"
    assert np.array_equal(disabled["cleaned_image"], image)
    assert marked["provenance"]["zodi_action"] == "not_subtracted"
    assert any("already removed" in warning for warning in marked["warnings"])


def test_fits_boundary_returns_cleaning_products_and_uses_cache(tmp_path: Path):
    path = tmp_path / "science.fits"
    header = _header()
    image = np.full((32, 32), 10.0, dtype=np.float32)
    image[16, 16] = 50.0
    flags = np.zeros(image.shape, dtype=np.uint32)
    flags[5, 5] = 1 << 25
    variance = np.ones_like(image)
    zodi = np.full_like(image, 2.0)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image, header=header, name="IMAGE"),
            fits.ImageHDU(flags, name="FLAGS"),
            fits.ImageHDU(variance, name="VARIANCE"),
            fits.ImageHDU(zodi, name="ZODI"),
        ]
    ).writeto(path)
    processor = SpherexFitsProcessor(tmp_path / "previews")

    first = processor.extract_target(path, 150.0, 2.0, size=20)
    second = processor.extract_target(path, 150.0, 2.0, size=20)

    assert first["data"].shape == first["original_data"].shape == (20, 20)
    assert first["quality_mask"].shape == (20, 20)
    assert first["artefact_mask"].shape == (20, 20)
    assert first["background_model"].shape == (20, 20)
    assert first["metadata"]["cleaning"]["cached"] is False
    assert second["metadata"]["cleaning"]["cached"] is True
    assert second["metadata"]["cleaning"]["provenance"]["pipeline"] == "spherex-cleaner-1"


def test_variance_and_unknown_flags_are_diagnostic_not_blanket_nonzero_mask():
    image = np.ones((8, 8))
    variance = np.ones_like(image)
    variance[0, 0] = 0
    flags = np.zeros(image.shape, dtype=np.uint64)
    flags[1, 1] = 1 << 5  # intentionally undocumented
    result = SpherexImageCleaner().clean(image, flags=flags, variance=variance)
    assert result["quality_mask"][0, 0]
    assert not result["quality_mask"][1, 1]
    assert result["statistics"]["unknown_flag_value_or"] == 1 << 5
