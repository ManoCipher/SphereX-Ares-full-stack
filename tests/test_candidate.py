import math
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from spherex.blink import align_frames, image_pixel_to_png, png_to_image_pixel, render_blink
from spherex.candidate import format_report, investigate_candidate, measure_source

PIXEL_ARCSEC = 6.0
RA0, DEC0 = 150.25, 2.1


def _wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [31.0, 31.0]
    wcs.wcs.crval = [RA0, DEC0]
    wcs.wcs.cdelt = [-PIXEL_ARCSEC / 3600.0, PIXEL_ARCSEC / 3600.0]
    return wcs


def _star(x, y, flux=200.0, sigma=1.2, shape=(60, 60)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))


def _frames(mover_positions, dates, seed=1):
    rng = np.random.default_rng(seed)
    frames = []
    for (mx, my), date in zip(mover_positions, dates, strict=True):
        data = rng.normal(0.0, 1.0, (60, 60)) + _star(15, 40, flux=300.0)  # a fixed star
        data += _star(mx, my)
        frames.append({"data": data, "wcs": _wcs(), "observation_id": date, "observation_timestamp": date,
                       "detector": "6", "wavelength_um": 1.0})
    return align_frames(frames)


def test_measure_source_centroid_and_non_detection():
    data = np.random.default_rng(0).normal(0, 1, (40, 40)) + _star(20.3, 18.6, shape=(40, 40))
    found = measure_source(data, 21, 18)
    assert found["found"] and found["snr"] > 50
    assert found["x"] == pytest.approx(20.3, abs=0.1) and found["y"] == pytest.approx(18.6, abs=0.1)
    empty = measure_source(np.random.default_rng(0).normal(0, 1, (40, 40)), 20, 20)
    assert empty["found"] is False and "sigma" in empty["reason"]


def test_moving_source_is_reported_with_rate_and_direction():
    # Moves 2 px (12") north over one year, in three steps.
    aligned = _frames([(30, 30), (30, 31), (30, 32)], ["2025-08-20", "2026-02-18", "2026-08-20"])
    report = investigate_candidate(aligned, 30.4, 30.2, clicked_frame=0)
    assert report["verdict"] == "Moving candidate"
    assert report["detected_frames"] == 3
    assert report["displacement_arcsec"] == pytest.approx(12.0, abs=1.0)
    assert report["motion_arcsec_per_year"] == pytest.approx(12.0, abs=1.5)
    assert report["motion_north_arcsec_per_year"] > 10 and abs(report["motion_east_arcsec_per_year"]) < 1.5
    assert report["position_angle_deg"] == pytest.approx(0.0, abs=8.0) or report["position_angle_deg"] > 352
    text = format_report(report)
    assert "Moving candidate" in text and "CANDIDATE INVESTIGATION REPORT" in text


def test_fixed_star_is_not_a_candidate():
    aligned = _frames([(30, 30), (30, 30)], ["2025-08-20", "2026-08-20"])
    report = investigate_candidate(aligned, 15, 40)  # the fixed star
    assert report["verdict"] == "No significant motion"
    assert report["displacement_arcsec"] < 1.0


def test_source_missing_in_frames_and_bad_frame_index():
    aligned = _frames([(30, 30), (55, 5)], ["2025-08-20", "2026-08-20"])  # second jumps far away
    report = investigate_candidate(aligned, 30, 30)
    assert report["verdict"] == "Not enough detections"
    assert report["detected_frames"] == 1
    assert "not found" in format_report(report)
    with pytest.raises(ValueError):
        investigate_candidate(aligned, 30, 30, clicked_frame=5)


def test_png_click_maps_back_to_the_same_image_pixel(tmp_path: Path):
    aligned = _frames([(30, 30), (30, 31)], ["2025-08-20", "2026-08-20"])
    result = render_blink(aligned, tmp_path)
    mapping = result["pixel_mapping"]
    column, row = image_pixel_to_png(mapping, 30.0, 30.0)
    assert 0 < column < 500 and 0 < row < 460  # inside the 500x460 PNG
    x, y = png_to_image_pixel(mapping, column, row)
    assert (x, y) == pytest.approx((30.0, 30.0))
    # Higher image y (north) is higher up the PNG (smaller row).
    assert image_pixel_to_png(mapping, 30.0, 40.0)[1] < row
    assert result["aligned"] is aligned and not math.isnan(mapping["x_scale"])
