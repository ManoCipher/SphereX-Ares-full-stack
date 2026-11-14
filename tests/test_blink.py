from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from spherex.blink import (
    BlinkError,
    align_frames,
    apply_scale,
    difference_image,
    prepare_blink,
    render_blink,
    shared_scale,
    wavelength_check,
)

RA0, DEC0 = 150.25, 2.1
PIXEL_DEG = 6.0 / 3600.0


def _wcs(crpix=(31.0, 31.0)) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.crval = [RA0, DEC0]
    wcs.wcs.cdelt = [-PIXEL_DEG, PIXEL_DEG]
    return wcs


def _star(shape, x, y, flux=100.0, sigma=1.5):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))


def _frame(timestamp, crpix, *, mover_offset=0.0, background=10.0, wavelength=2.0, detector="4"):
    """60x60 frame: a fixed star at the sky centre, a mover east of it."""
    shape = (60, 60)
    wcs = _wcs(crpix)
    centre_x, centre_y = (value - 1.0 for value in crpix)  # FITS crpix is 1-based
    data = background + _star(shape, centre_x, centre_y)
    data += _star(shape, centre_x + 10 + mover_offset, centre_y, flux=60.0)
    return {
        "data": data,
        "wcs": wcs,
        "observation_id": f"obs-{timestamp}",
        "observation_timestamp": timestamp,
        "detector": detector,
        "wavelength_um": wavelength,
    }


def _peak(image, region):
    sub = np.nan_to_num(image[region], nan=-np.inf)
    y, x = np.unravel_index(np.argmax(sub), sub.shape)
    return x + region[1].start, y + region[0].start


def test_alignment_puts_fixed_star_on_same_pixel():
    # Same sky, but the target sits at a different place in each cutout.
    early = _frame("2025-05-06T12:00:00", (31.0, 31.0))
    late = _frame("2025-10-31T17:00:00", (34.0, 28.0))
    aligned = align_frames([late, early])  # deliberately out of order

    assert aligned["reference_observation_id"] == "obs-2025-05-06T12:00:00"
    region = (slice(25, 36), slice(25, 36))
    first, second = (_peak(frame, region) for frame in aligned["frames"])
    assert first == second == (30, 30)
    assert aligned["overlap_fraction"] > 0.8


def test_difference_cancels_fixed_star_and_keeps_mover():
    early = _frame("2025-05-06", (31.0, 31.0))
    late = _frame("2025-10-31", (33.0, 31.0), mover_offset=4.0)
    aligned = align_frames([early, late])
    diff = difference_image(aligned)

    fixed = np.nanmax(np.abs(diff[26:35, 26:35]))
    mover = np.nanmax(np.abs(diff[25:36, 36:48]))
    assert fixed < 0.1 * mover
    assert np.nanmax(diff) > 20 and np.nanmin(diff) < -20  # +/- pair of the moving star


def test_background_is_removed_per_frame():
    early = _frame("2025-05-06", (31.0, 31.0), background=10.0)
    late = _frame("2025-10-31", (31.0, 31.0), background=55.0)
    aligned = align_frames([early, late])
    medians = [float(np.nanmedian(frame)) for frame in aligned["frames"]]
    assert medians == pytest.approx([0.0, 0.0], abs=1e-6)
    assert aligned["backgrounds"][1] - aligned["backgrounds"][0] == pytest.approx(45.0, abs=0.5)


def test_shared_scale_uses_all_frames_and_apply_scale_keeps_nan():
    low, high = shared_scale([np.array([0.0, 1.0]), np.array([np.nan, 100.0])], percentiles=(0.0, 100.0))
    assert (low, high) == (0.0, 100.0)
    scaled = apply_scale(np.array([0.0, 50.0, np.nan]), 0.0, 100.0, "linear")
    assert scaled[0] == 0.0 and scaled[1] == pytest.approx(0.5) and np.isnan(scaled[2])
    with pytest.raises(ValueError):
        apply_scale(np.zeros(2), 0.0, 1.0, "log")


def test_wavelength_check():
    same = wavelength_check([{"wavelength_um": 2.00, "detector": "4"}, {"wavelength_um": 2.02, "detector": "4"}])
    assert same["comparable"] is True
    different = wavelength_check([{"wavelength_um": 2.0, "detector": "4"}, {"wavelength_um": 2.3, "detector": "2"}])
    assert different["comparable"] is False
    assert different["detectors"] == ["2", "4"]
    unknown = wavelength_check([{"wavelength_um": 2.0}, {"wavelength_um": None}])
    assert unknown["comparable"] is None


def test_rejects_bad_input():
    good = _frame("2025-05-06", (31.0, 31.0))
    with pytest.raises(BlinkError, match="2 to 6"):
        align_frames([good])
    with pytest.raises(BlinkError, match="WCS"):
        align_frames([good, {**good, "wcs": None}])
    far = _frame("2025-10-31", (31.0, 31.0))
    far_wcs = _wcs()
    far_wcs.wcs.crval = [RA0 + 1.0, DEC0]
    with pytest.raises(BlinkError, match="common sky coverage"):
        align_frames([good, {**far, "wcs": far_wcs}])


def test_render_blink_writes_frames_and_difference(tmp_path: Path):
    aligned = align_frames([_frame("2025-05-06", (31.0, 31.0)), _frame("2025-10-31", (32.0, 30.0), mover_offset=3.0)])
    result = render_blink(aligned, tmp_path, target_ra=RA0, target_dec=DEC0, colormap="Inferno")
    assert result["status"] == "success"
    assert len(result["frame_paths"]) == 2
    assert all(Path(path).is_file() for path in result["frame_paths"])
    assert Path(result["difference_path"]).is_file()
    with pytest.raises(ValueError):
        render_blink(aligned, tmp_path, colormap="rainbow")


class _FakeProcessor:
    def __init__(self, frames):
        self.frames = frames

    def extract_target(self, path, ra, dec, size):
        frame = self.frames[Path(path).name]
        return {"data": frame["data"], "wcs": frame["wcs"], "metadata": {"pixel_x": 30.0, "pixel_y": 30.0}}

    def wavelength_at_target(self, path, x, y, record):
        return {"wavelength_um": self.frames[Path(path).name]["wavelength_um"], "source": "test"}


class _FakeService:
    def __init__(self, frames, missing=()):
        self.processor = _FakeProcessor(frames)
        self.missing = set(missing)
        self.calls: list[int] = []

    def resolve_fits_path(self, observation):
        return None  # pretend nothing is local yet, so every frame is "downloaded"

    def refresh_files(self):
        pass

    def ensure_local_files(self, observations, ra, dec):
        self.calls.append(len(observations))
        records = []
        for observation in observations:
            record = dict(observation)
            if record["observation_id"] in self.missing:
                record["download_error"] = "network down"
            else:
                record["file_path"] = f"{record['observation_id']}.fits"
            records.append(record)
        return records


def test_prepare_blink_end_to_end_with_fake_service(tmp_path: Path):
    frames = {
        "A.fits": _frame("2025-05-06", (31.0, 31.0), wavelength=2.0),
        "B.fits": _frame("2025-10-31", (32.0, 31.0), wavelength=2.5, detector="4"),
    }
    observations = [
        {"observation_id": "B", "observation_timestamp": "2025-10-31", "detector": "4"},
        {"observation_id": "A", "observation_timestamp": "2025-05-06", "detector": "4"},
    ]
    service = _FakeService(frames)
    messages: list[str] = []
    result = prepare_blink(service, observations, RA0, DEC0, tmp_path, progress=messages.append)
    # One single-frame request per missing frame (parallel), then one final pass over all.
    assert sorted(service.calls) == [1, 1, 2]
    assert any("at a time" in message for message in messages)
    assert [meta["observation_id"] for meta in result["metadata"]] == ["A", "B"]
    assert result["wavelength"]["comparable"] is False
    assert len(result["frame_paths"]) == 2

    assert result["skipped"] == []

    # One unusable frame out of three is skipped, not fatal.
    frames["C.fits"] = _frame("2026-05-01", (31.0, 31.0))
    three = observations + [{"observation_id": "C", "observation_timestamp": "2026-05-01", "detector": "4"}]
    partial = prepare_blink(_FakeService(frames, missing={"A"}), three, RA0, DEC0, tmp_path)
    assert [meta["observation_id"] for meta in partial["metadata"]] == ["B", "C"]
    assert "network down" in partial["skipped"][0]

    # Fewer than two usable frames is an error that explains why.
    with pytest.raises(BlinkError, match="network down"):
        prepare_blink(_FakeService(frames, missing={"A"}), observations, RA0, DEC0, tmp_path)
    with pytest.raises(BlinkError, match="2 to 6"):
        prepare_blink(_FakeService(frames), observations[:1], RA0, DEC0, tmp_path)
