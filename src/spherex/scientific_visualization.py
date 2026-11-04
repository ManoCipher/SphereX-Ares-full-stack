from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import FITSFixedWarning, WCS

from .targets import Target

LOG = logging.getLogger(__name__)

# These bit names are taken from the official definitions copied into every
# SPHEREx FLAGS header.  MP_SOURCE is informational and MP_FULLSAMPLE indicates
# extra data, so neither is rejected.  Unknown non-zero bits are reported but
# are not silently treated as fatal.
REJECT_FLAG_NAMES = {
    "TRANSIENT",
    "OVERFLOW",
    "SUR_ERROR",
    "PHANTOM",
    "REFERENCE",
    "NONFUNC",
    "DICHROIC",
    "MISSING_DATA",
    "HOT",
    "COLD",
    "PHANMISS",
    "NONLINEAR",
    "PERSIST",
    "OUTLIER",
}


def observation_time(header: fits.Header, fallback_mjd: float | None = None) -> Time:
    """Return the scientifically meaningful exposure midpoint when available."""
    for key in ("MJD-AVG", "MJD-OBS", "MJD-BEG"):
        value = header.get(key)
        if value is not None:
            return Time(float(value), format="mjd", scale=str(header.get("TIMESYS", "UTC")).lower())
    for key in ("DATE-AVG", "DATE-OBS", "DATE-BEG"):
        value = header.get(key)
        if value:
            return Time(str(value), format="isot", scale=str(header.get("TIMESYS", "UTC")).lower())
    if fallback_mjd is not None and np.isfinite(fallback_mjd):
        return Time(float(fallback_mjd), format="mjd", scale="utc")
    raise ValueError("No observation timestamp in IMAGE header or verified metadata")


def spacecraft_offset(header: fits.Header) -> CartesianRepresentation | None:
    """Read the exposure-midpoint geocentric spacecraft position, if defined."""
    if "GEOCENTER" not in str(header.get("XYZ_SC_SYSTEM", "")).upper():
        return None
    values = [header.get(key) for key in ("X_SC", "Y_SC", "Z_SC")]
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    return CartesianRepresentation(*(float(value) * u.km for value in values))


def pixel_is_inside(x: float, y: float, shape: tuple[int, int]) -> bool:
    """Test a continuous, zero-based coordinate against the pixel footprint."""
    ny, nx = shape
    return bool(np.isfinite(x) and np.isfinite(y) and -0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5)


def target_pixel(
    header: fits.Header, shape: tuple[int, int], target: Target, time: Time
) -> tuple[Any, float, float, bool]:
    """Propagate a target and map its observer-dependent ICRS position to pixels."""
    coord = target.at_observer(time, spacecraft_offset(header))
    wcs = WCS(header).celestial
    x, y = wcs.world_to_pixel(coord)
    x, y = float(x), float(y)
    return coord, x, y, pixel_is_inside(x, y, shape)


def flag_definitions(header: fits.Header) -> dict[str, int]:
    definitions: dict[str, int] = {}
    for card in header.cards:
        key = str(card.keyword)
        if key.startswith("MP_"):
            try:
                definitions[key[3:]] = int(card.value)
            except (TypeError, ValueError):
                continue
    return definitions


def quality_mask(flags: np.ndarray, header: fits.Header) -> tuple[np.ndarray, list[str]]:
    definitions = flag_definitions(header)
    reject_bits = 0
    used: list[str] = []
    for name in sorted(REJECT_FLAG_NAMES):
        if name in definitions:
            reject_bits |= 1 << definitions[name]
            used.append(name)
    values = np.asarray(flags, dtype=np.int64)
    return (values & reject_bits) != 0, used


def wavelength_at(hdul: fits.HDUList, x: float, y: float) -> tuple[float, float]:
    """Evaluate the alternate -TAB wavelength WCS at a cutout pixel."""
    if "WCS-WAVE" not in hdul:
        raise ValueError("Missing WCS-WAVE extension")
    spectral_header = hdul["IMAGE"].header.copy()
    # wcslib otherwise notices the primary celestial SIP coefficients while
    # constructing alternate WCS W and emits a misleading inconsistency log.
    for key in list(spectral_header):
        if key in {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"} or re.fullmatch(r"(?:A|B|AP|BP)_\d+_\d+", key):
            del spectral_header[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        spectral_wcs = WCS(spectral_header, fobj=hdul, key="W")
    # The archive tutorial explicitly disables SIP for detector-filter
    # wavelength coordinates, which are independent of optical distortion.
    spectral_wcs.sip = None
    wavelength, bandwidth = spectral_wcs.pixel_to_world(float(x), float(y))
    return float(wavelength.to_value(u.um)), float(bandwidth.to_value(u.um))


def _aperture_weights(shape: tuple[int, int], x: float, y: float, radius: float, samples: int = 16) -> np.ndarray:
    """Fractional circular-aperture weights from deterministic subpixel sampling."""
    yy, xx = np.indices(shape, dtype=float)
    offsets = (np.arange(samples, dtype=float) + 0.5) / samples - 0.5
    weights = np.zeros(shape, dtype=float)
    for dy in offsets:
        for dx in offsets:
            weights += ((xx + dx - x) ** 2 + (yy + dy - y) ** 2 <= radius**2)
    return weights / samples**2


def aperture_photometry(
    image_minus_zodi: np.ndarray,
    variance: np.ndarray,
    invalid: np.ndarray,
    x: float,
    y: float,
    aperture_radius: float = 1.5,
    annulus_inner: float = 2.5,
    annulus_outer: float = 5.0,
) -> dict[str, float | str | int]:
    """Conservative forced-aperture significance with a local annulus.

    This is not a per-pixel IMAGE/sigma ratio.  It subtracts a robust local
    background and propagates both the aperture variance and uncertainty in the
    background estimate.  It assumes independent pixels; therefore even a high
    value remains a candidate requiring calibrated forced photometry.
    """
    signal = np.asarray(image_minus_zodi, dtype=float)
    var = np.asarray(variance, dtype=float)
    bad = np.asarray(invalid, dtype=bool) | ~np.isfinite(signal) | ~np.isfinite(var) | (var <= 0)
    yy, xx = np.indices(signal.shape, dtype=float)
    distance = np.hypot(xx - x, yy - y)
    annulus = (distance >= annulus_inner) & (distance <= annulus_outer) & ~bad
    background_values = signal[annulus]
    result: dict[str, float | str | int] = {
        "local_background": np.nan,
        "measurement_uncertainty": np.nan,
        "aperture_snr": np.nan,
        "aperture_signal": np.nan,
        "background_pixels": int(background_values.size),
        "aperture_bad_fraction": np.nan,
        "status": "INCONCLUSIVE",
    }
    if background_values.size < 8:
        result["reason"] = "insufficient valid local-background pixels"
        return result

    background = float(np.nanmedian(background_values))
    mad_sigma = 1.4826 * float(np.nanmedian(np.abs(background_values - background)))
    median_sigma = float(np.sqrt(np.nanmedian(var[annulus])))
    scatter = max(mad_sigma, median_sigma)
    background_uncertainty = 1.2533 * scatter / np.sqrt(background_values.size)

    weights = _aperture_weights(signal.shape, x, y, aperture_radius)
    total_weight = float(weights.sum())
    valid_weights = np.where(bad, 0.0, weights)
    good_weight = float(valid_weights.sum())
    bad_fraction = 1.0 - good_weight / total_weight if total_weight else 1.0
    result["local_background"] = background
    result["aperture_bad_fraction"] = bad_fraction
    if total_weight == 0 or good_weight / total_weight < 0.8:
        result["reason"] = "more than 20% of target aperture is invalid or off-image"
        return result

    aperture_signal = float(np.sum(valid_weights * (signal - background)))
    aperture_variance = float(np.sum(valid_weights**2 * var))
    aperture_variance += (good_weight * background_uncertainty) ** 2
    if not np.isfinite(aperture_variance) or aperture_variance <= 0:
        result["reason"] = "non-positive propagated aperture variance"
        return result
    uncertainty = float(np.sqrt(aperture_variance))
    snr = aperture_signal / uncertainty
    result.update(
        aperture_signal=aperture_signal,
        measurement_uncertainty=uncertainty,
        aperture_snr=float(snr),
        status="EVIDENCE" if snr >= 5.0 else "NOT DETECTED",
        reason=(
            "forced-aperture SNR >= 5; candidate evidence, not a catalog-quality detection"
            if snr >= 5.0
            else "forced-aperture SNR < 5"
        ),
    )
    return result


def _release_from_path(path: Path, header: fits.Header) -> str:
    match = re.search(r"(?:^|[_-])(qr[23])(?:[_-]|$)", path.name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    value = str(header.get("DATAREL", header.get("RELEASE", "UNKNOWN"))).upper()
    return value


def _plot_image(ax: Any, data: np.ndarray, title: str, unit: str, cmap: str = "viridis", symmetric: bool = False) -> None:
    finite = np.asarray(data, dtype=float)
    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), 98)) if np.isfinite(finite).any() else 1.0
        limit = limit if limit > 0 else 1.0
        image = ax.imshow(finite, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
    else:
        norm = ImageNormalize(finite, interval=PercentileInterval(99), stretch=AsinhStretch())
        image = ax.imshow(finite, origin="lower", cmap=cmap, norm=norm)
    ax.set(title=title, xlabel="Pixel X (0-based)", ylabel="Pixel Y (0-based)")
    plt.colorbar(image, ax=ax, label=unit, fraction=0.046, pad=0.04)


def _mark_target(ax: Any, x: float, y: float, inside: bool) -> None:
    if inside:
        ax.plot(x, y, marker="+", color="red", markersize=18, markeredgewidth=2.2, label="Predicted WISE 0855")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)


def analyze_fits(
    path: Path,
    plot_dir: Path,
    target: Target,
    fallback_mjd: float | None = None,
    make_plot: bool = True,
) -> dict[str, Any]:
    """Analyze one unchanged SPHEREx cutout and optionally write its dashboard."""
    warning_messages: list[str] = []
    row: dict[str, Any] = {
        "filename": path.name,
        "observation_id": "",
        "timestamp": "",
        "detector": np.nan,
        "data_release": "UNKNOWN",
        "target_ra_deg": np.nan,
        "target_dec_deg": np.nan,
        "target_x": np.nan,
        "target_y": np.nan,
        "target_inside": False,
        "position_status": "UNAVAILABLE",
        "central_wavelength_um": np.nan,
        "bandwidth_um": np.nan,
        "peak_pixel_mjy_sr": np.nan,
        "local_background_mjy_sr": np.nan,
        "measurement_uncertainty_mjy_sr": np.nan,
        "aperture_snr": np.nan,
        "flagged_pixel_count": np.nan,
        "rejected_pixel_count": np.nan,
        "target_flag_names": "",
        "detection_status": "INCONCLUSIVE",
        "scientific_use": "INCONCLUSIVE",
        "warnings": "",
        "dashboard": "",
    }
    with fits.open(path, memmap=False) as hdul:
        if "IMAGE" not in hdul:
            raise ValueError("Missing IMAGE extension")
        image_hdu = hdul["IMAGE"]
        image = np.asarray(image_hdu.data, dtype=float)
        if image.ndim != 2:
            raise ValueError(f"IMAGE must be 2-D, got shape {image.shape}")
        header = image_hdu.header
        row.update(
            observation_id=str(header.get("OBSID", path.stem)),
            detector=header.get("DETECTOR", np.nan),
            data_release=_release_from_path(path, header),
        )

        try:
            obstime = observation_time(header, fallback_mjd)
            row["timestamp"] = obstime.utc.isot
        except (ValueError, TypeError) as exc:
            obstime = None
            warning_messages.append(str(exc))

        inside = False
        x = y = np.nan
        coord = None
        if obstime is not None:
            try:
                coord, x, y, inside = target_pixel(header, image.shape, target, obstime)
                row.update(
                    target_ra_deg=coord.ra.deg,
                    target_dec_deg=coord.dec.deg,
                    target_x=x,
                    target_y=y,
                    target_inside=inside,
                    position_status="INSIDE" if inside else "TARGET OUTSIDE CUTOUT",
                )
            except Exception as exc:  # malformed or unsupported FITS WCS
                warning_messages.append(f"spatial WCS: {exc}")

        if inside:
            try:
                wavelength, bandwidth = wavelength_at(hdul, x, y)
                row.update(central_wavelength_um=wavelength, bandwidth_um=bandwidth)
            except Exception as exc:
                warning_messages.append(f"spectral WCS: {exc}")
        elif np.isfinite(x) and np.isfinite(y):
            warning_messages.append("wavelength not evaluated outside the observed cutout")

        required = [name for name in ("VARIANCE", "FLAGS", "ZODI") if name not in hdul]
        if required:
            warning_messages.append("missing extension(s): " + ", ".join(required))

        variance = np.asarray(hdul["VARIANCE"].data, dtype=float) if "VARIANCE" in hdul else np.full_like(image, np.nan)
        zodi = np.asarray(hdul["ZODI"].data, dtype=float) if "ZODI" in hdul else np.full_like(image, np.nan)
        if "FLAGS" in hdul:
            flags = np.asarray(hdul["FLAGS"].data)
            rejected, rejected_names = quality_mask(flags, hdul["FLAGS"].header)
            row["flagged_pixel_count"] = int(np.count_nonzero(flags))
            row["rejected_pixel_count"] = int(np.count_nonzero(rejected))
            if inside:
                nearest_x = int(np.clip(np.rint(x), 0, image.shape[1] - 1))
                nearest_y = int(np.clip(np.rint(y), 0, image.shape[0] - 1))
                target_flag = int(flags[nearest_y, nearest_x])
                definitions = flag_definitions(hdul["FLAGS"].header)
                names = [name for name, bit in definitions.items() if target_flag & (1 << bit)]
                row["target_flag_names"] = "|".join(names) if names else "NONE"
            if not rejected_names:
                warning_messages.append("FLAGS has no recognized official MP_* rejection bits")
        else:
            flags = np.zeros(image.shape, dtype=np.int64)
            rejected = np.ones(image.shape, dtype=bool)

        if variance.shape != image.shape or zodi.shape != image.shape or rejected.shape != image.shape:
            warning_messages.append("IMAGE/VARIANCE/FLAGS/ZODI shapes do not match")
            variance = np.full_like(image, np.nan)
            zodi = np.full_like(image, np.nan)
            rejected = np.ones(image.shape, dtype=bool)

        invalid_variance = ~np.isfinite(variance) | (variance <= 0)
        if np.count_nonzero(invalid_variance):
            warning_messages.append(f"{np.count_nonzero(invalid_variance)} invalid variance pixel(s)")
        invalid = rejected | invalid_variance | ~np.isfinite(image) | ~np.isfinite(zodi)
        diagnostic = image - zodi

        photometry: dict[str, Any] = {"local_background": np.nan, "measurement_uncertainty": np.nan, "aperture_snr": np.nan, "status": "INCONCLUSIVE"}
        if inside and not required:
            photometry = aperture_photometry(diagnostic, variance, invalid, x, y)
            aperture = _aperture_weights(image.shape, x, y, 1.5) > 0
            valid_aperture = aperture & ~invalid
            if np.any(valid_aperture):
                row["peak_pixel_mjy_sr"] = float(np.nanmax(image[valid_aperture]))
            row.update(
                local_background_mjy_sr=photometry["local_background"],
                measurement_uncertainty_mjy_sr=photometry["measurement_uncertainty"],
                aperture_snr=photometry["aperture_snr"],
                detection_status=photometry["status"],
            )
            if photometry.get("reason"):
                warning_messages.append(str(photometry["reason"]))
        elif not inside and row["position_status"] == "TARGET OUTSIDE CUTOUT":
            row["detection_status"] = "INCONCLUSIVE"
            warning_messages.append("target outside cutout; no target photometry")

        complete = inside and not required and np.isfinite(row["central_wavelength_um"])
        if complete and row["detection_status"] in {"NOT DETECTED", "EVIDENCE"}:
            row["scientific_use"] = "USEFUL DIAGNOSTIC; precision work requires CWAVE/CBAND and calibrated forced photometry"
        elif not inside:
            row["scientific_use"] = "NOT USEFUL FOR TARGET: target outside cutout"
        else:
            row["scientific_use"] = "INCONCLUSIVE: incomplete or invalid calibration information"

        if make_plot:
            plot_dir.mkdir(parents=True, exist_ok=True)
            dashboard = plot_dir / f"{path.stem}_science_dashboard.png"
            valid_map = ~invalid
            background_for_map = float(photometry.get("local_background", np.nan))
            if not np.isfinite(background_for_map) and np.any(valid_map):
                background_for_map = float(np.nanmedian(diagnostic[valid_map]))
            with np.errstate(divide="ignore", invalid="ignore"):
                residual_snr = (diagnostic - background_for_map) / np.sqrt(variance)
            residual_snr[~valid_map] = np.nan

            fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
            _plot_image(axes[0, 0], image, "Original IMAGE", str(header.get("BUNIT", "MJy/sr")))
            _plot_image(axes[0, 1], np.where(invalid, np.nan, diagnostic), "IMAGE − ZODI (diagnostic)", "MJy/sr")
            _plot_image(
                axes[1, 0],
                residual_snr,
                "Per-pixel standardized residual\n(not source-detection significance)",
                "residual / sqrt(VARIANCE)",
                cmap="coolwarm",
                symmetric=True,
            )
            for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
                _mark_target(ax, x, y, inside)

            axes[1, 1].axis("off")
            position_line = (
                f"Pixel: ({x:.2f}, {y:.2f}) — INSIDE"
                if inside
                else ("TARGET OUTSIDE CUTOUT" if np.isfinite(x) else "Target position unavailable")
            )
            wavelength_line = (
                f"λ = {row['central_wavelength_um']:.4f} µm; Δλ = {row['bandwidth_um']:.4f} µm"
                if np.isfinite(row["central_wavelength_um"])
                else "Wavelength unavailable at target"
            )
            snr_line = f"Target aperture SNR: {row['aperture_snr']:.2f}" if np.isfinite(row["aperture_snr"]) else "Target aperture SNR: invalid"
            metadata_text = "\n".join(
                [
                    "SCIENTIFIC STATUS",
                    f"{target.name}",
                    f"Predicted ICRS: {row['target_ra_deg']:.8f}°, {row['target_dec_deg']:.8f}°" if coord is not None else "Predicted ICRS unavailable",
                    position_line,
                    f"Observation: {row['timestamp'] or 'unknown'} UTC",
                    f"Observation ID: {row['observation_id']}",
                    f"Detector D{row['detector']} | {row['data_release']}",
                    wavelength_line,
                    "WCS-WAVE is visualization-grade; use CWAVE/CBAND for precision.",
                    "",
                    f"Peak target-aperture pixel: {row['peak_pixel_mjy_sr']:.5g} MJy/sr" if np.isfinite(row["peak_pixel_mjy_sr"]) else "Peak target-aperture pixel: unavailable",
                    f"Local background after ZODI subtraction: {row['local_background_mjy_sr']:.5g} MJy/sr" if np.isfinite(row["local_background_mjy_sr"]) else "Local background: unavailable",
                    f"Aperture measurement uncertainty: {row['measurement_uncertainty_mjy_sr']:.5g} MJy/sr" if np.isfinite(row["measurement_uncertainty_mjy_sr"]) else "Measurement uncertainty: unavailable",
                    snr_line,
                    f"Flagged pixels: {row['flagged_pixel_count']} (rejected: {row['rejected_pixel_count']})",
                    f"Nearest target-pixel flags: {row['target_flag_names'] or 'unavailable'}",
                    f"Detection assessment: {row['detection_status']}",
                    "A bright pixel alone is not identification of WISE 0855.",
                ]
            )
            axes[1, 1].text(0.01, 0.99, metadata_text, va="top", ha="left", fontsize=10, linespacing=1.35)
            fig.suptitle(f"SPHEREx WISE 0855 scientific diagnostic — {path.name}", fontsize=13)
            fig.savefig(dashboard, dpi=160)
            plt.close(fig)
            row["dashboard"] = str(dashboard)

    row["warnings"] = "; ".join(dict.fromkeys(warning_messages))
    return row


def analyze_directory(
    raw_dir: Path,
    plot_dir: Path,
    summary_path: Path,
    target: Target,
    metadata_path: Path | None = None,
) -> pd.DataFrame:
    """Analyze every existing FITS cutout without any network activity."""
    fallback_times: dict[tuple[str, int], float] = {}
    if metadata_path is not None and metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        for record in metadata.to_dict("records"):
            detector_match = re.search(r"D([1-6])(?:_|$)", str(record.get("access_url", "")))
            if detector_match and pd.notna(record.get("obs_time_mjd")):
                fallback_times[(str(record.get("obs_id")), int(detector_match.group(1)))] = float(record["obs_time_mjd"])

    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.fits")):
        fallback = None
        try:
            with fits.open(path, memmap=False) as hdul:
                if "IMAGE" in hdul:
                    h = hdul["IMAGE"].header
                    fallback = fallback_times.get((str(h.get("OBSID", "")), int(h.get("DETECTOR", -1))))
            rows.append(analyze_fits(path, plot_dir, target, fallback_mjd=fallback))
        except Exception as exc:
            LOG.exception("Could not analyze %s", path)
            rows.append(
                {
                    "filename": path.name,
                    "position_status": "UNAVAILABLE",
                    "detection_status": "INCONCLUSIVE",
                    "scientific_use": "INCONCLUSIVE: processing failure",
                    "warnings": str(exc),
                }
            )
    frame = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary_path, index=False)
    return frame
