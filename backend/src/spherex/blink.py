"""WCS-aligned blink comparison of SPHEREx cutouts taken at different times.

Every frame is reprojected onto the sky grid of one reference frame, so fixed
stars land on the same pixels and only real changes (motion, variability)
remain when the frames are flipped. Unlike the RGB composite, all frames share
ONE brightness scale; otherwise the blink would show display differences
instead of sky differences.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.wcs import WCS

from .image_renderer import ALLOWED_COLORMAPS, ALLOWED_STRETCHES

MIN_FRAMES = 2
MAX_FRAMES = 6
MIN_COMMON_PIXELS = 25
MIN_OVERLAP_FRACTION = 0.01
# SPHEREx resolving power is about 40-130, so lambda/R is roughly 1-2.5 %.
# Frames further apart than this are not "the same wavelength".
WAVELENGTH_TOLERANCE_FRACTION = 0.02
ASINH_SOFTNESS = 10.0
MAX_PARALLEL_DOWNLOADS = 3


class BlinkError(RuntimeError):
    """Raised for blink failures whose message is safe to show in the UI."""


def _timestamp_key(frame: Mapping[str, Any]) -> str:
    return str(frame.get("observation_timestamp") or "")


def _validate_frames(frames: Sequence[Mapping[str, Any]]) -> None:
    if not MIN_FRAMES <= len(frames) <= MAX_FRAMES:
        raise BlinkError(f"Blink needs {MIN_FRAMES} to {MAX_FRAMES} frames; got {len(frames)}")
    for index, frame in enumerate(frames):
        data = np.asarray(frame.get("data"), dtype=float)
        if data.ndim != 2 or not data.size:
            raise BlinkError(f"Frame {index + 1} is not a two-dimensional image")
        wcs = frame.get("wcs")
        if not isinstance(wcs, WCS) or not wcs.has_celestial:
            raise BlinkError(f"Frame {index + 1} has no valid celestial WCS")


def _reproject(data: np.ndarray, wcs: WCS, reference_wcs: WCS, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    try:
        from reproject import reproject_interp
    except ImportError as exc:
        raise BlinkError("Blink requires the 'reproject' package; install project dependencies again") from exc
    try:
        projected, footprint = reproject_interp((data, wcs), reference_wcs, shape_out=shape, return_footprint=True)
    except Exception as exc:
        raise BlinkError(f"Could not align frame to the reference sky grid: {exc}") from exc
    return np.asarray(projected, dtype=float), np.asarray(footprint, dtype=float)


def wavelength_check(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Say whether the frames see the target at a comparable wavelength."""
    values = [frame.get("wavelength_um") for frame in frames]
    known = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    detectors = sorted({str(frame.get("detector")) for frame in frames if frame.get("detector") not in (None, "")})
    if len(known) != len(frames):
        return {
            "comparable": None,
            "spread_um": None,
            "spread_fraction": None,
            "detectors": detectors,
            "note": "Wavelength at the target is unknown for some frames; brightness changes may be spectral.",
        }
    spread = max(known) - min(known)
    fraction = spread / (sum(known) / len(known))
    comparable = fraction <= WAVELENGTH_TOLERANCE_FRACTION
    note = (
        f"Wavelengths at the target agree within {fraction:.1%}."
        if comparable
        else f"Wavelengths at the target differ by {fraction:.1%} ({spread:.3f} um): "
        "brightness differences may be spectral, not changes in time."
    )
    return {
        "comparable": comparable,
        "spread_um": spread,
        "spread_fraction": fraction,
        "detectors": detectors,
        "note": note,
    }


def align_frames(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sort frames by time and reproject all of them onto the earliest one.

    Each frame mapping needs ``data`` (2-D array) and ``wcs`` (celestial WCS);
    other keys (observation_id, observation_timestamp, detector,
    wavelength_um, ...) are carried through as metadata. Input arrays are not
    modified.
    """
    _validate_frames(frames)
    ordered = sorted(frames, key=_timestamp_key)
    reference = ordered[0]
    reference_wcs: WCS = reference["wcs"]
    reference_data = np.array(reference["data"], dtype=float, copy=True)
    shape = reference_data.shape

    aligned: list[np.ndarray] = [reference_data]
    masks: list[np.ndarray] = [np.isfinite(reference_data)]
    for frame in ordered[1:]:
        projected, footprint = _reproject(np.asarray(frame["data"], dtype=float), frame["wcs"], reference_wcs, shape)
        aligned.append(projected)
        masks.append(np.isfinite(projected) & (footprint > 0.0))

    common_mask = np.logical_and.reduce(masks)
    common_pixels = int(common_mask.sum())
    overlap_fraction = float(common_pixels / common_mask.size)
    if common_pixels < MIN_COMMON_PIXELS or overlap_fraction < MIN_OVERLAP_FRACTION:
        raise BlinkError(f"Selected frames have insufficient common sky coverage ({overlap_fraction:.2%})")

    # Zodiacal light and detector offsets change between visits; removing each
    # frame's median background keeps the blink about sources, not about sky level.
    backgrounds = [float(np.median(frame[common_mask])) for frame in aligned]
    flattened = [np.where(common_mask, frame - background, np.nan) for frame, background in zip(aligned, backgrounds, strict=True)]

    metadata = [
        {key: value for key, value in frame.items() if key not in {"data", "wcs"}}
        for frame in ordered
    ]
    return {
        "frames": flattened,
        "wcs": reference_wcs,
        "common_mask": common_mask,
        "backgrounds": backgrounds,
        "overlap_fraction": overlap_fraction,
        "common_valid_pixels": common_pixels,
        "reference_observation_id": reference.get("observation_id"),
        "metadata": metadata,
        "wavelength": wavelength_check(ordered),
    }


def shared_scale(frames: Sequence[np.ndarray], percentiles: tuple[float, float] = (2.0, 98.0)) -> tuple[float, float]:
    """One display range computed from the pixels of ALL frames together."""
    finite = np.concatenate([frame[np.isfinite(frame)] for frame in frames])
    if not finite.size:
        raise BlinkError("Aligned frames contain no finite pixels")
    low, high = (float(value) for value in np.percentile(finite, percentiles))
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            high = low + 1.0
    return low, high


def apply_scale(data: np.ndarray, low: float, high: float, stretch: str = "asinh") -> np.ndarray:
    """Map data to 0..1 with a fixed range; NaN stays NaN (drawn as background)."""
    stretch_name = stretch.casefold()
    if stretch_name not in ALLOWED_STRETCHES:
        raise ValueError(f"Unsupported brightness stretch: {stretch}")
    scaled = np.clip((np.asarray(data, dtype=float) - low) / (high - low), 0.0, 1.0)
    if stretch_name == "asinh":
        scaled = np.arcsinh(ASINH_SOFTNESS * scaled) / np.arcsinh(ASINH_SOFTNESS)
    return scaled


def difference_image(aligned: Mapping[str, Any], first: int = 0, last: int = -1) -> np.ndarray:
    """Later minus earlier frame: fixed stars cancel, a mover leaves a +/- pair."""
    frames = aligned["frames"]
    return np.asarray(frames[last], dtype=float) - np.asarray(frames[first], dtype=float)


def _safe_name(value: Any, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback))[:60]


def _draw(
    image: np.ndarray,
    wcs: WCS,
    destination: Path,
    title: str,
    *,
    colormap: str,
    vmin: float,
    vmax: float,
    target: tuple[float, float] | None,
    show_grid: bool,
) -> dict[str, float]:
    """Save the PNG and return how image pixels map onto PNG pixels.

    The layout is fixed (no tight bounding box), so every frame of one blink
    has the same mapping and a click on the PNG can be turned back into an
    image pixel, and from there into sky coordinates.
    """
    figure = plt.figure(figsize=(5.0, 4.6), dpi=100)
    figure.subplots_adjust(left=0.2, right=0.96, bottom=0.12, top=0.93)
    try:
        axes = figure.add_subplot(111, projection=wcs)
        axes.imshow(image, origin="lower", cmap=colormap, vmin=vmin, vmax=vmax)
        axes.coords[0].set_axislabel("Right Ascension (deg)")
        axes.coords[1].set_axislabel("Declination (deg)")
        axes.coords[0].set_format_unit(u.deg)
        axes.coords[1].set_format_unit(u.deg)
        if show_grid:
            axes.coords.grid(color="white", alpha=0.35, linestyle=":", linewidth=0.8)
        if target is not None:
            axes.plot(
                [target[0]], [target[1]], marker="o", markersize=16, markerfacecolor="none",
                markeredgewidth=1.4, color="cyan", transform=axes.get_transform("world"),
            )
        axes.set_title(title, fontsize=9)
        figure.canvas.draw()
        (x0, y0), (x1, y1) = axes.transData.transform([(0.0, 0.0), (1.0, 1.0)])
        png_height = figure.get_size_inches()[1] * figure.dpi
        figure.savefig(destination, facecolor="white")
        # PNG column = x_scale * image_x + x_offset; PNG row = y_scale * image_y + y_offset
        # (image y grows upwards, PNG rows grow downwards, hence the flip).
        return {
            "x_scale": float(x1 - x0),
            "x_offset": float(x0),
            "y_scale": float(-(y1 - y0)),
            "y_offset": float(png_height - y0),
        }
    finally:
        plt.close(figure)


def png_to_image_pixel(mapping: Mapping[str, float], column: float, row: float) -> tuple[float, float]:
    """Invert the mapping returned by the renderer: PNG click -> image (x, y)."""
    return (
        (column - mapping["x_offset"]) / mapping["x_scale"],
        (row - mapping["y_offset"]) / mapping["y_scale"],
    )


def image_pixel_to_png(mapping: Mapping[str, float], x: float, y: float) -> tuple[float, float]:
    return x * mapping["x_scale"] + mapping["x_offset"], y * mapping["y_scale"] + mapping["y_offset"]


def render_blink(
    aligned: Mapping[str, Any],
    output_dir: Path | str,
    *,
    target_ra: float | None = None,
    target_dec: float | None = None,
    colormap: str = "inferno",
    stretch: str = "asinh",
    show_grid: bool = True,
) -> dict[str, Any]:
    """Write one PNG per aligned frame (shared scale) plus a difference PNG.

    Every PNG has the same pixel grid and axes, so a viewer can flip between
    them without anything but the sky changing.
    """
    colormap_name = colormap.casefold()
    if colormap_name not in ALLOWED_COLORMAPS:
        raise ValueError(f"Unsupported colour palette: {colormap}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    wcs: WCS = aligned["wcs"]
    frames: list[np.ndarray] = aligned["frames"]
    target = (float(target_ra), float(target_dec)) if target_ra is not None and target_dec is not None else None
    low, high = shared_scale(frames)

    frame_paths: list[str] = []
    try:
        for index, (frame, meta) in enumerate(zip(frames, aligned["metadata"], strict=True)):
            name = _safe_name(meta.get("observation_id"), f"frame{index}")
            path = destination / f"blink_{index:02d}_{name}_D{_safe_name(meta.get('detector'), 'x')}.png"
            title = f"{meta.get('observation_timestamp') or 'date unknown'}  |  D{meta.get('detector') or '-'}"
            mapping = _draw(apply_scale(frame, low, high, stretch), wcs, path, title, colormap=colormap_name,
                            vmin=0.0, vmax=1.0, target=target, show_grid=show_grid)
            frame_paths.append(str(path.resolve()))

        difference = difference_image(aligned)
        finite = np.abs(difference[np.isfinite(difference)])
        limit = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        limit = limit if math.isfinite(limit) and limit > 0 else 1.0
        difference_path = destination / "blink_difference.png"
        _draw(difference, wcs, difference_path, "Last minus first (white = brighter later)",
              colormap="gray", vmin=-limit, vmax=limit, target=target, show_grid=show_grid)
    except (OSError, ValueError) as exc:
        raise BlinkError(f"Blink PNG generation failed: {exc}") from exc

    return {
        "status": "success",
        "frame_paths": frame_paths,
        "difference_path": str(difference_path.resolve()),
        "scale_low": low,
        "scale_high": high,
        "stretch": stretch.casefold(),
        "colormap": colormap_name,
        "overlap_fraction": aligned["overlap_fraction"],
        "reference_observation_id": aligned["reference_observation_id"],
        "metadata": aligned["metadata"],
        "wavelength": aligned["wavelength"],
        # For clicking on the PNGs (candidate investigation); same for every frame.
        "pixel_mapping": mapping,
        "aligned": aligned,
    }


def fetch_in_parallel(
    service: Any,
    observations: Sequence[Mapping[str, Any]],
    ra: float,
    dec: float,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Make every selected FITS local, downloading the missing ones at the same time.

    IRSA needs roughly 25 s to cut out one SPHEREx frame, and the shared
    downloader fetches one file after another. Asking for each missing frame in
    its own thread turns N waits into about one. Order of the result matches
    the input.
    """
    report = progress or (lambda _message: None)
    records = [dict(observation) for observation in observations]
    missing = [index for index, record in enumerate(records) if service.resolve_fits_path(record) is None]
    if not missing:
        return service.ensure_local_files(records, ra, dec)
    report(f"Downloading {len(missing)} of {len(records)} frame(s) from IRSA, {MAX_PARALLEL_DOWNLOADS} at a time (about 30-90 s)...")

    def fetch(index: int) -> tuple[int, dict[str, Any]]:
        return index, service.ensure_local_files([records[index]], ra, dec)[0]

    # More than ~3 simultaneous cutout requests makes IRSA answer 503 (seen on 14 Nov 2026).
    workers = min(MAX_PARALLEL_DOWNLOADS, len(missing))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="blink-download") as pool:
        futures = [pool.submit(fetch, index) for index in missing]
        for done, future in enumerate(as_completed(futures), start=1):
            index, record = future.result()
            records[index] = record
            report(f"Downloaded {done} of {len(missing)} missing frame(s)...")
    service.refresh_files()
    # Re-resolve everything (already-local ones included) through the normal path.
    return service.ensure_local_files(records, ra, dec)


def prepare_blink(
    service: Any,
    observations: Sequence[Mapping[str, Any]],
    ra: float,
    dec: float,
    output_dir: Path | str,
    *,
    size: int = 200,
    colormap: str = "inferno",
    stretch: str = "asinh",
    show_grid: bool = True,
    show_target: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Download (if needed), cut out, align and render the selected observations.

    ``service`` is an ``ObservationPreviewService``; only its public methods
    and its ``processor`` are used, so the existing preview code stays unchanged.
    ``progress`` receives short status messages (called from this thread).
    """
    if not MIN_FRAMES <= len(observations) <= MAX_FRAMES:
        raise BlinkError(f"Select {MIN_FRAMES} to {MAX_FRAMES} observations to blink; got {len(observations)}")
    report = progress or (lambda _message: None)
    records = fetch_in_parallel(service, observations, ra, dec, report)
    report("Aligning frames on the sky grid and drawing them...")
    processor = service.processor
    frames: list[dict[str, Any]] = []
    skipped: list[str] = []
    for record in records:
        label = record.get("observation_id") or record.get("obs_id") or "observation"
        if not record.get("file_path"):
            skipped.append(f"{label}: {record.get('download_error') or 'FITS file is not available'}")
            continue
        path = Path(record["file_path"])
        try:
            cutout = processor.extract_target(path, ra, dec, size)
        except Exception as exc:  # e.g. target just off the detector edge
            skipped.append(f"{label}: {exc}")
            continue
        metadata = cutout["metadata"]
        wavelength = processor.wavelength_at_target(path, metadata["pixel_x"], metadata["pixel_y"], record)
        frames.append(
            {
                "data": cutout["data"],
                "wcs": cutout["wcs"],
                "observation_id": label,
                "observation_timestamp": record.get("observation_timestamp") or metadata.get("observation_timestamp"),
                "detector": record.get("detector") or metadata.get("detector"),
                "band": record.get("band"),
                "wavelength_um": wavelength["wavelength_um"],
                "wavelength_source": wavelength["source"],
                "fits_filename": path.name,
            }
        )
    if len(frames) < MIN_FRAMES:
        raise BlinkError(f"Only {len(frames)} usable frame(s). " + " | ".join(skipped))
    aligned = align_frames(frames)
    result = render_blink(
        aligned,
        output_dir,
        target_ra=ra if show_target else None,
        target_dec=dec if show_target else None,
        colormap=colormap,
        stretch=stretch,
        show_grid=show_grid,
    )
    return {**result, "skipped": skipped}
