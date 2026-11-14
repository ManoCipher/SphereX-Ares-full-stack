from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from spherex.image_renderer import SpherexImageRenderer


def _wcs(crval=(150.25, 2.1), crpix=(31.0, 31.0)) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.crval = list(crval)
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    return wcs


def test_false_colour_and_wcs_rendering(tmp_path: Path):
    renderer = SpherexImageRenderer()
    data = np.arange(3600, dtype=float).reshape(60, 60)
    data[:3, :] = np.nan
    cutout = {"data": data, "wcs": _wcs()}

    false_colour = renderer.render_false_color(
        cutout, tmp_path / "inferno.png", colormap="inferno", stretch="asinh"
    )
    wcs_image = renderer.render_wcs_image(
        cutout,
        150.25,
        2.1,
        tmp_path / "wcs.png",
        colormap="viridis",
        stretch="linear",
        show_grid=True,
        show_target=True,
    )

    assert Path(false_colour["preview_path"]).is_file()
    assert false_colour["colormap"] == "inferno"
    assert false_colour["stretch"] == "asinh"
    assert Path(wcs_image["preview_path"]).is_file()
    assert wcs_image["wcs_axes"] is True
    assert wcs_image["show_grid"] is True
    assert wcs_image["show_target"] is True


def test_rgb_composite_reprojects_and_orders_channels_by_wavelength(tmp_path: Path):
    renderer = SpherexImageRenderer()
    y, x = np.mgrid[:60, :60]
    base = np.exp(-((x - 30) ** 2 + (y - 30) ** 2) / 80.0)
    inputs = [
        {
            "data": base * 2,
            "wcs": _wcs(crpix=(30.5, 31.0)),
            "observation_id": "LONG",
            "detector": "6",
            "wavelength_um": 4.8,
            "wavelength_source": "test",
        },
        {
            "data": base,
            "wcs": _wcs(),
            "observation_id": "SHORT",
            "detector": "1",
            "wavelength_um": 0.9,
            "wavelength_source": "test",
        },
        {
            "data": base * 1.5,
            "wcs": _wcs(crpix=(31.0, 30.5)),
            "observation_id": "MID",
            "detector": "3",
            "wavelength_um": 2.2,
            "wavelength_source": "test",
        },
    ]
    result = renderer.create_rgb_composite(
        inputs, 150.25, 2.1, tmp_path / "rgb.png", show_grid=True, show_target=True
    )

    assert result["status"] == "success"
    assert result["mapping_method"] == "wavelength_ordered"
    assert result["channels"]["B"]["observation_id"] == "SHORT"
    assert result["channels"]["G"]["observation_id"] == "MID"
    assert result["channels"]["R"]["observation_id"] == "LONG"
    assert result["overlap_fraction"] > 0.9
    assert Path(result["preview_path"]).is_file()


def test_manual_rgb_mapping_is_explicit_when_wavelength_is_missing(tmp_path: Path):
    renderer = SpherexImageRenderer()
    data = np.ones((30, 30))
    inputs = [
        {"data": data * value, "wcs": _wcs(crpix=(16, 16)), "observation_id": name, "wavelength_um": None}
        for value, name in enumerate(("FIRST", "SECOND", "THIRD"), start=1)
    ]
    result = renderer.create_rgb_composite(inputs, 150.25, 2.1, tmp_path / "manual.png")
    assert result["mapping_method"] == "manual_selection_order"
    assert result["channels"]["R"]["observation_id"] == "FIRST"
    assert "selection order" in result["mapping_note"]
