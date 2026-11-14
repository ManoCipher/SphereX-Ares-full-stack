"""Astronomical display rendering for SPHEREx cutouts and RGB composites."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.wcs import WCS


ALLOWED_COLORMAPS = {"gray", "inferno", "viridis", "magma", "plasma"}
ALLOWED_STRETCHES = {"linear", "asinh"}


class ImageRenderingError(RuntimeError):
    """Raised for display or reprojection failures safe to show in the UI."""


class SpherexImageRenderer:
    """Render display products without changing their scientific input arrays."""

    @staticmethod
    def normalize(data: np.ndarray, stretch: str = "asinh") -> tuple[np.ndarray, dict[str, Any]]:
        stretch_name = stretch.casefold()
        if stretch_name not in ALLOWED_STRETCHES:
            raise ValueError(f"Unsupported brightness stretch: {stretch}")
        values = np.asarray(data, dtype=float)
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        if not finite.size:
            raise ImageRenderingError("Image contains no finite pixels")
        low, high = (float(value) for value in np.percentile(finite, (2.0, 98.0)))
        constant = not (math.isfinite(low) and math.isfinite(high)) or high <= low
        if constant:
            low = float(np.min(finite))
            high = float(np.max(finite))
            if high <= low:
                high = low + 1.0
        scaled = np.zeros(values.shape, dtype=np.float32)
        scaled[finite_mask] = np.clip((values[finite_mask] - low) / (high - low), 0.0, 1.0)
        if stretch_name == "asinh":
            softness = 10.0
            scaled[finite_mask] = np.arcsinh(softness * scaled[finite_mask]) / np.arcsinh(softness)
        return scaled, {
            "scale_low": low,
            "scale_high": high,
            "scale_percentiles": [2.0, 98.0],
            "stretch": stretch_name,
            "constant_image": constant,
            "finite_pixel_count": int(finite.size),
        }

    @staticmethod
    def _validate_colormap(colormap: str) -> str:
        name = colormap.casefold().replace("greyscale", "gray")
        if name not in ALLOWED_COLORMAPS:
            raise ValueError(f"Unsupported colour palette: {colormap}")
        return name

    def render_false_color(
        self,
        cutout: Mapping[str, Any],
        output_path: Path | str,
        *,
        colormap: str = "inferno",
        stretch: str = "asinh",
    ) -> dict[str, Any]:
        """Save a NaN-safe false-colour PNG without WCS decorations."""
        palette = self._validate_colormap(colormap)
        data = np.asarray(cutout["data"], dtype=float)
        normalized, scaling = self.normalize(data, stretch)
        rgba = plt.get_cmap(palette)(normalized)
        rgba[..., 3] = np.isfinite(data).astype(float)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.imsave(destination, rgba, origin="lower")
        except Exception as exc:
            raise ImageRenderingError(f"False-colour PNG generation failed: {exc}") from exc
        return {
            "preview_path": str(destination.resolve()),
            "colormap": palette,
            "wcs_axes": False,
            **scaling,
        }

    def render_wcs_image(
        self,
        cutout: Mapping[str, Any],
        target_ra: float,
        target_dec: float,
        output_path: Path | str,
        *,
        colormap: str = "inferno",
        stretch: str = "asinh",
        show_grid: bool = True,
        show_target: bool = True,
    ) -> dict[str, Any]:
        """Render the updated Cutout2D WCS with optional grid and target marker."""
        palette = self._validate_colormap(colormap)
        wcs = cutout.get("wcs")
        if not isinstance(wcs, WCS) or not wcs.has_celestial:
            raise ImageRenderingError("Cutout has no valid celestial WCS for coordinate axes")
        data = np.asarray(cutout["data"], dtype=float)
        normalized, scaling = self.normalize(data, stretch)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure = plt.figure(figsize=(5.0, 4.5), dpi=120)
            axes = figure.add_subplot(111, projection=wcs)
            axes.imshow(normalized, origin="lower", cmap=palette, vmin=0.0, vmax=1.0)
            axes.coords[0].set_axislabel("Right Ascension (deg)")
            axes.coords[1].set_axislabel("Declination (deg)")
            axes.coords[0].set_format_unit(u.deg)
            axes.coords[1].set_format_unit(u.deg)
            if show_grid:
                axes.coords.grid(color="white", alpha=0.38, linestyle=":", linewidth=0.8)
            if show_target:
                transform = axes.get_transform("world")
                axes.plot(
                    [target_ra],
                    [target_dec],
                    marker="+",
                    markersize=16,
                    markeredgewidth=2.0,
                    color="yellow",
                    transform=transform,
                )
                axes.text(
                    target_ra,
                    target_dec,
                    f"  Target\n  {target_ra:.6f}, {target_dec:.6f}",
                    color="yellow",
                    fontsize=7,
                    va="bottom",
                    transform=transform,
                )
            figure.tight_layout()
            figure.savefig(destination, facecolor="white", bbox_inches="tight")
            plt.close(figure)
        except Exception as exc:
            plt.close("all")
            raise ImageRenderingError(f"WCS preview generation failed: {exc}") from exc
        return {
            "preview_path": str(destination.resolve()),
            "colormap": palette,
            "wcs_axes": True,
            "show_grid": bool(show_grid),
            "show_target": bool(show_target),
            **scaling,
        }

    def create_rgb_composite(
        self,
        channel_inputs: Sequence[Mapping[str, Any]],
        target_ra: float,
        target_dec: float,
        output_path: Path | str,
        *,
        show_grid: bool = True,
        show_target: bool = True,
        stretch: str = "asinh",
    ) -> dict[str, Any]:
        """Reproject exactly three scientific cutouts to a common WCS grid."""
        if len(channel_inputs) != 3:
            raise ValueError("RGB composite generation requires exactly three observations")
        try:
            from reproject import reproject_interp
        except ImportError as exc:
            raise ImageRenderingError(
                "RGB composites require the 'reproject' package; install project dependencies again"
            ) from exc

        wavelengths = [entry.get("wavelength_um") for entry in channel_inputs]
        reliable = all(value is not None and math.isfinite(float(value)) for value in wavelengths)
        if reliable:
            ordered = sorted(channel_inputs, key=lambda entry: float(entry["wavelength_um"]))
            assignments = {"B": ordered[0], "G": ordered[1], "R": ordered[2]}
            mapping_method = "wavelength_ordered"
            mapping_note = "Shortest wavelength to blue; intermediate to green; longest to red."
        else:
            assignments = dict(zip(("R", "G", "B"), channel_inputs, strict=True))
            mapping_method = "manual_selection_order"
            mapping_note = "Wavelength metadata incomplete; selection order explicitly mapped to R, G, B."

        reference = assignments["G"]
        reference_wcs = reference.get("wcs")
        reference_data = np.asarray(reference.get("data"), dtype=float)
        if not isinstance(reference_wcs, WCS) or not reference_wcs.has_celestial:
            raise ImageRenderingError("Green reference channel has no valid celestial WCS")
        shape_out = reference_data.shape
        if reference_data.ndim != 2 or not all(dimension > 0 for dimension in shape_out):
            raise ImageRenderingError("RGB reference channel is not a two-dimensional image")

        aligned: dict[str, np.ndarray] = {}
        valid_masks: list[np.ndarray] = []
        for channel, entry in assignments.items():
            data = np.asarray(entry.get("data"), dtype=float)
            wcs = entry.get("wcs")
            if not isinstance(wcs, WCS) or not wcs.has_celestial:
                raise ImageRenderingError(f"{channel} channel has no valid celestial WCS")
            try:
                if entry is reference:
                    projected = np.array(data, dtype=float, copy=True)
                    footprint = np.isfinite(projected).astype(float)
                else:
                    projected, footprint = reproject_interp(
                        (data, wcs), reference_wcs, shape_out=shape_out, return_footprint=True
                    )
            except Exception as exc:
                raise ImageRenderingError(f"Could not align {channel} channel: {exc}") from exc
            aligned[channel] = projected
            valid_masks.append(np.isfinite(projected) & (footprint > 0.0))

        common_mask = np.logical_and.reduce(valid_masks)
        common_pixels = int(common_mask.sum())
        overlap_fraction = float(common_pixels / common_mask.size)
        if common_pixels < 25 or overlap_fraction < 0.01:
            raise ImageRenderingError(
                f"Selected observations have insufficient common sky coverage ({overlap_fraction:.2%})"
            )

        normalized: dict[str, np.ndarray] = {}
        scaling: dict[str, dict[str, Any]] = {}
        for channel in ("R", "G", "B"):
            channel_data = np.where(common_mask, aligned[channel], np.nan)
            normalized[channel], scaling[channel] = self.normalize(channel_data, stretch)
        rgb = np.dstack((normalized["R"], normalized["G"], normalized["B"]))
        rgb[~common_mask] = 0.0

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure = plt.figure(figsize=(7.0, 6.0), dpi=120)
            axes = figure.add_subplot(111, projection=reference_wcs)
            axes.imshow(rgb, origin="lower")
            axes.coords[0].set_axislabel("Right Ascension (deg)")
            axes.coords[1].set_axislabel("Declination (deg)")
            axes.coords[0].set_format_unit(u.deg)
            axes.coords[1].set_format_unit(u.deg)
            if show_grid:
                axes.coords.grid(color="white", alpha=0.35, linestyle=":", linewidth=0.8)
            if show_target:
                transform = axes.get_transform("world")
                axes.plot(
                    [target_ra], [target_dec], marker="+", markersize=18,
                    markeredgewidth=2.2, color="yellow", transform=transform,
                )
            axes.set_title("Representative false-colour SPHEREx infrared composite")
            figure.tight_layout()
            figure.savefig(destination, facecolor="white", bbox_inches="tight")
            plt.close(figure)
        except Exception as exc:
            plt.close("all")
            raise ImageRenderingError(f"RGB composite PNG generation failed: {exc}") from exc

        channel_metadata = {
            channel: {
                "observation_id": entry.get("observation_id"),
                "detector": entry.get("detector"),
                "band": entry.get("band"),
                "wavelength_um": entry.get("wavelength_um"),
                "wavelength_source": entry.get("wavelength_source"),
                "scaling": scaling[channel],
            }
            for channel, entry in assignments.items()
        }
        return {
            "status": "success",
            "preview_path": str(destination.resolve()),
            "target_ra": float(target_ra),
            "target_dec": float(target_dec),
            "image_width": int(shape_out[1]),
            "image_height": int(shape_out[0]),
            "overlap_fraction": overlap_fraction,
            "common_valid_pixels": common_pixels,
            "mapping_method": mapping_method,
            "mapping_note": mapping_note,
            "channels": channel_metadata,
            "show_grid": bool(show_grid),
            "show_target": bool(show_target),
            "stretch": stretch.casefold(),
            "label": "Representative false-colour infrared composite",
            "calibration_note": (
                "Channels are independently normalized for visual comparison; "
                "their displayed brightnesses are not a calibrated spectral measurement."
            ),
        }
