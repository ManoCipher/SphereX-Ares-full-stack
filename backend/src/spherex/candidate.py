"""Candidate investigation: measure a clicked source in every blink frame.

Flow: the user clicks a suspected moving object in one aligned frame. The
click becomes an image pixel and, through the shared WCS, a sky position. The
source is then located near that position in every frame (peak search, then a
flux-weighted centroid), converted to RA/Dec, and the angular displacement
between frames is measured and compared with the measurement uncertainty.

All frames of a blink are already reprojected onto one sky grid with their
median background removed (see ``blink.align_frames``), so pixel positions
are directly comparable between frames.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .time_selection import observation_mjd

SEARCH_RADIUS_PX = 4.0  # ~25 arcsec for SPHEREx: covers a year of Barnard's Star motion
CENTROID_HALF_WIDTH = 2  # 5x5 pixel centroid box
DETECTION_SNR = 5.0
PSF_FWHM_PX = 1.5  # rough SPHEREx PSF width in pixels, only used for the error estimate
# Reprojection and centroiding are not better than this, whatever the signal.
POSITION_FLOOR_PX = 0.15
SIGNIFICANCE_THRESHOLD = 3.0
DAYS_PER_YEAR = 365.25
OFFSET_HEADER = 'offset"'  # kept out of the f-string: Python 3.11 forbids quotes/backslashes there


def noise_sigma(frame: np.ndarray) -> float:
    """Robust pixel noise (MAD scaled to a Gaussian sigma)."""
    finite = frame[np.isfinite(frame)]
    if not finite.size:
        return math.nan
    mad = float(np.median(np.abs(finite - np.median(finite))))
    return 1.4826 * mad if mad > 0 else float(np.std(finite)) or 1.0


def measure_source(
    frame: np.ndarray,
    x: float,
    y: float,
    *,
    radius: float = SEARCH_RADIUS_PX,
    min_snr: float = DETECTION_SNR,
) -> dict[str, Any]:
    """Find the brightest source within ``radius`` pixels of (x, y) and centroid it."""
    data = np.asarray(frame, dtype=float)
    height, width = data.shape
    sigma = noise_sigma(data)
    yy, xx = np.mgrid[:height, :width]
    inside = ((xx - x) ** 2 + (yy - y) ** 2 <= radius**2) & np.isfinite(data)
    if not inside.any() or not math.isfinite(sigma) or sigma <= 0:
        return {"found": False, "reason": "no valid pixels near the position", "snr": None}
    candidates = np.where(inside, data, -np.inf)
    peak_y, peak_x = np.unravel_index(int(np.argmax(candidates)), data.shape)
    peak = float(data[peak_y, peak_x])
    snr = peak / sigma
    if snr < min_snr:
        return {"found": False, "reason": f"no source above {min_snr:g} sigma (best {snr:.1f})", "snr": snr}

    y0, y1 = max(0, peak_y - CENTROID_HALF_WIDTH), min(height, peak_y + CENTROID_HALF_WIDTH + 1)
    x0, x1 = max(0, peak_x - CENTROID_HALF_WIDTH), min(width, peak_x + CENTROID_HALF_WIDTH + 1)
    box = np.nan_to_num(data[y0:y1, x0:x1], nan=0.0)
    weights = np.clip(box, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        cx, cy = float(peak_x), float(peak_y)
    else:
        by, bx = np.mgrid[y0:y1, x0:x1]
        cx, cy = float((weights * bx).sum() / total), float((weights * by).sum() / total)
    error_px = max(POSITION_FLOOR_PX, PSF_FWHM_PX / (2.355 * snr))
    return {"found": True, "x": cx, "y": cy, "peak": peak, "snr": snr, "position_error_px": error_px}


def _pixel_scale_arcsec(wcs: WCS) -> float:
    scales = proj_plane_pixel_scales(wcs.celestial) * 3600.0  # degrees -> arcsec
    return float(np.mean(scales))


def _sky(wcs: WCS, x: float, y: float) -> SkyCoord:
    return wcs.celestial.pixel_to_world(x, y)


def _fit_motion(times_yr: np.ndarray, east: np.ndarray, north: np.ndarray) -> tuple[float, float]:
    """Least-squares linear motion (arcsec/yr) in east and north."""
    if len(times_yr) < 2 or np.ptp(times_yr) <= 0:
        return math.nan, math.nan
    design = np.vstack([times_yr - times_yr.mean(), np.ones_like(times_yr)]).T
    mu_east = float(np.linalg.lstsq(design, east, rcond=None)[0][0])
    mu_north = float(np.linalg.lstsq(design, north, rcond=None)[0][0])
    return mu_east, mu_north


def investigate_candidate(
    aligned: Mapping[str, Any],
    click_x: float,
    click_y: float,
    *,
    clicked_frame: int = 0,
    radius: float = SEARCH_RADIUS_PX,
) -> dict[str, Any]:
    """Measure the source near a clicked pixel in all aligned frames and report its motion."""
    frames: Sequence[np.ndarray] = aligned["frames"]
    metadata: Sequence[Mapping[str, Any]] = aligned["metadata"]
    wcs: WCS = aligned["wcs"]
    if not 0 <= clicked_frame < len(frames):
        raise ValueError(f"clicked_frame must be between 0 and {len(frames) - 1}")
    scale = _pixel_scale_arcsec(wcs)
    click_sky = _sky(wcs, click_x, click_y)

    # Lock on to the source in the clicked frame first, then look around it everywhere.
    anchor = measure_source(frames[clicked_frame], click_x, click_y, radius=radius)
    centre_x, centre_y = (anchor["x"], anchor["y"]) if anchor["found"] else (click_x, click_y)

    rows: list[dict[str, Any]] = []
    for index, (frame, meta) in enumerate(zip(frames, metadata, strict=True)):
        measurement = measure_source(frame, centre_x, centre_y, radius=radius)
        row: dict[str, Any] = {
            "frame": index + 1,
            "observation_id": meta.get("observation_id"),
            "observation_timestamp": meta.get("observation_timestamp"),
            "mjd": observation_mjd(meta),
            "detector": meta.get("detector"),
            "wavelength_um": meta.get("wavelength_um"),
            **measurement,
        }
        if measurement["found"]:
            sky = _sky(wcs, measurement["x"], measurement["y"])
            row.update(ra=float(sky.ra.deg), dec=float(sky.dec.deg), position_error_arcsec=measurement["position_error_px"] * scale)
        rows.append(row)

    found = [row for row in rows if row["found"] and row["mjd"] is not None]
    report: dict[str, Any] = {
        "click": {"x": click_x, "y": click_y, "ra": float(click_sky.ra.deg), "dec": float(click_sky.dec.deg),
                  "frame": clicked_frame + 1},
        "pixel_scale_arcsec": scale,
        "search_radius_arcsec": radius * scale,
        "measurements": rows,
        "detected_frames": len(found),
        "total_frames": len(rows),
    }
    if len(found) < 2:
        report.update(verdict="Not enough detections",
                      explanation="The source was found in fewer than two dated frames, so no motion can be measured.")
        return report

    found.sort(key=lambda row: row["mjd"])
    first, last = found[0], found[-1]
    origin = SkyCoord(first["ra"] * u.deg, first["dec"] * u.deg)
    east, north = [], []
    for row in found:
        d_east, d_north = origin.spherical_offsets_to(SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg))
        row["offset_east_arcsec"] = float(d_east.to_value(u.arcsec))
        row["offset_north_arcsec"] = float(d_north.to_value(u.arcsec))
        row["offset_arcsec"] = math.hypot(row["offset_east_arcsec"], row["offset_north_arcsec"])
        east.append(row["offset_east_arcsec"])
        north.append(row["offset_north_arcsec"])

    span_days = last["mjd"] - first["mjd"]
    displacement = last["offset_arcsec"]
    displacement_error = math.hypot(first["position_error_arcsec"], last["position_error_arcsec"])
    significance = displacement / displacement_error if displacement_error > 0 else math.inf
    times_yr = np.array([(row["mjd"] - first["mjd"]) / DAYS_PER_YEAR for row in found])
    mu_east, mu_north = _fit_motion(times_yr, np.array(east), np.array(north))
    rate = math.hypot(mu_east, mu_north) if math.isfinite(mu_east) else math.nan
    position_angle = math.degrees(math.atan2(mu_east, mu_north)) % 360 if math.isfinite(rate) else math.nan

    moving = significance >= SIGNIFICANCE_THRESHOLD and span_days > 0
    report.update(
        time_span_days=span_days,
        displacement_arcsec=displacement,
        displacement_pixels=displacement / scale,
        displacement_error_arcsec=displacement_error,
        significance_sigma=significance,
        motion_arcsec_per_year=rate,
        motion_east_arcsec_per_year=mu_east,
        motion_north_arcsec_per_year=mu_north,
        position_angle_deg=position_angle,
        verdict="Moving candidate" if moving else "No significant motion",
        explanation=(
            f"Moved {displacement:.1f}\" ({displacement / scale:.2f} px) in {span_days:.0f} days, "
            f"{significance:.1f} sigma above the position uncertainty."
            if moving
            else f"Shift of {displacement:.1f}\" is within {SIGNIFICANCE_THRESHOLD:g} sigma of the position "
            f"uncertainty ({displacement_error:.1f}\"): consistent with a fixed source."
        ),
    )
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """Plain-text candidate investigation report for the UI or a file."""
    click = report["click"]
    lines = [
        "CANDIDATE INVESTIGATION REPORT",
        "",
        f"Clicked in frame {click['frame']}:  RA {click['ra']:.6f}  Dec {click['dec']:+.6f}  (deg, ICRS)",
        f"Search radius {report['search_radius_arcsec']:.1f}\"  |  pixel scale {report['pixel_scale_arcsec']:.2f}\"",
        f"Detected in {report['detected_frames']} of {report['total_frames']} frames",
        "",
        f"{'#':>2}  {'Date (UTC)':<19}  {'Det':>3}  {'lambda um':>9}  {'RA deg':>11}  {'Dec deg':>10}  "
        f"{'SNR':>7}  {OFFSET_HEADER:>7}",
    ]
    for row in report["measurements"]:
        date = str(row.get("observation_timestamp") or "?")[:19]
        wavelength = row.get("wavelength_um")
        wavelength_text = f"{wavelength:9.3f}" if isinstance(wavelength, (int, float)) else f"{'?':>9}"
        if row["found"]:
            offset = row.get("offset_arcsec")
            offset_text = f"{offset:7.2f}" if offset is not None else f"{'-':>7}"
            lines.append(
                f"{row['frame']:>2}  {date:<19}  {str(row.get('detector') or '-'):>3}  {wavelength_text}  "
                f"{row['ra']:11.6f}  {row['dec']:+10.6f}  {row['snr']:7.0f}  {offset_text}"
            )
        else:
            lines.append(f"{row['frame']:>2}  {date:<19}  {str(row.get('detector') or '-'):>3}  {wavelength_text}  "
                         f"not found: {row.get('reason')}")
    lines += ["", f"VERDICT: {report['verdict']}", report["explanation"]]
    if "motion_arcsec_per_year" in report and math.isfinite(report["motion_arcsec_per_year"]):
        lines.append(
            f"Motion: {report['motion_arcsec_per_year']:.1f}\"/yr  (east {report['motion_east_arcsec_per_year']:+.1f}, "
            f"north {report['motion_north_arcsec_per_year']:+.1f})  position angle {report['position_angle_deg']:.0f} deg"
        )
    lines += [
        "",
        "Notes: positions are flux-weighted centroids on the aligned grid; uncertainty is an estimate",
        f"(PSF / SNR, at least {POSITION_FLOOR_PX} px). Shifts below one pixel are at the limit of SPHEREx",
        "resolution; frames at different wavelengths can also shift a centroid of a blended source.",
    ]
    return "\n".join(lines)
