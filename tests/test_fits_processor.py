from pathlib import Path

import numpy as np
from astropy.io import fits

from spherex.fits_processor import inspect_directory


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
