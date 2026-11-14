"""Backend FITS inspection, WCS cutouts, and PNG previews."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import FITSFixedWarning, WCS

from .image_renderer import ImageRenderingError, SpherexImageRenderer

LOG = logging.getLogger(__name__)


class FitsProcessingError(RuntimeError):
    """Base class for FITS errors that are safe to show in the UI."""


class InvalidWcsError(FitsProcessingError):
    """Raised when an image has no usable celestial WCS."""


class TargetOutsideImageError(FitsProcessingError):
    """Raised when the requested coordinate is outside the image footprint."""


class InvalidImageDataError(FitsProcessingError):
    """Raised when an image contains no usable numerical pixels."""


def classify_extension(extname: str) -> str:
    name = extname.upper()
    if any(token in name for token in ("VAR", "UNC", "ERROR", "SIGMA")):
        return "variance_or_uncertainty"
    if any(token in name for token in ("MASK", "FLAG", "DQ")):
        return "mask_or_quality"
    if "PSF" in name:
        return "psf"
    if any(token in name for token in ("WAVE", "LAMBDA")):
        return "wavelength_calibration"
    if "ZODI" in name:
        return "background_model"
    if any(token in name for token in ("SCI", "IMAGE", "FLUX")):
        return "science_candidate"
    return "unknown"


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _first_header(headers: Sequence[fits.Header], *keys: str) -> Any:
    for key in keys:
        for header in headers:
            value = header.get(key)
            if value not in (None, ""):
                return _plain(value)
    return None


def _as_2d(data: Any) -> np.ndarray | None:
    if not isinstance(data, np.ndarray) or not np.issubdtype(data.dtype, np.number):
        return None
    squeezed = np.squeeze(data)
    return squeezed if squeezed.ndim == 2 and squeezed.size else None


class SpherexFitsProcessor:
    """Process local FITS data without knowledge of any user interface."""

    def __init__(self, preview_dir: Path | str = "cache/previews") -> None:
        self.preview_dir = Path(preview_dir)
        self.renderer = SpherexImageRenderer()

    @staticmethod
    def _science_hdu_index(hdul: fits.HDUList) -> int:
        candidates: list[tuple[int, int]] = []
        for index, hdu in enumerate(hdul):
            if _as_2d(hdu.data) is None:
                continue
            name = str(hdu.header.get("EXTNAME", "PRIMARY" if index == 0 else "")).upper()
            if name == "IMAGE":
                rank = 0
            elif name in {"SCI", "SCIENCE"}:
                rank = 1
            elif any(token in name for token in ("IMAGE", "SCI", "FLUX")):
                rank = 2
            elif index == 0:
                rank = 3
            else:
                rank = 4
            candidates.append((rank, index))
        if not candidates:
            raise InvalidImageDataError("FITS file has no two-dimensional numeric image HDU")
        return min(candidates)[1]

    def inspect_fits(self, file_path: Path | str) -> dict[str, Any]:
        """Return structured HDU and science-image metadata for ``file_path``."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"FITS file not found: {path}")
        try:
            with fits.open(path, mode="readonly", memmap=False) as hdul:
                science_index = self._science_hdu_index(hdul)
                science = hdul[science_index]
                image = _as_2d(science.data)
                assert image is not None
                headers = (science.header, hdul[0].header)
                hdu_rows: list[dict[str, Any]] = []
                for index, hdu in enumerate(hdul):
                    data = hdu.data
                    numeric = isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number)
                    hdu_rows.append(
                        {
                            "index": index,
                            "name": str(hdu.header.get("EXTNAME", "PRIMARY" if index == 0 else f"HDU{index}")),
                            "classification": classify_extension(str(hdu.header.get("EXTNAME", ""))),
                            "shape": list(data.shape) if isinstance(data, np.ndarray) else None,
                            "dtype": str(data.dtype) if isinstance(data, np.ndarray) else None,
                            "finite_fraction": (
                                float(np.isfinite(data).sum() / data.size)
                                if numeric and data.size
                                else None
                            ),
                            "brightness_unit": _plain(hdu.header.get("BUNIT")),
                            "has_celestial_wcs": self._has_celestial_wcs(hdu.header),
                        }
                    )
                return {
                    "file_path": str(path.resolve()),
                    "filename": path.name,
                    "hdu_count": len(hdul),
                    "science_hdu_index": science_index,
                    "science_hdu_name": str(science.header.get("EXTNAME", "PRIMARY")),
                    "image_width": int(image.shape[1]),
                    "image_height": int(image.shape[0]),
                    "dtype": str(image.dtype),
                    "observation_id": _first_header(headers, "OBSID", "OBS_ID", "OBSID_L2", "EXP_ID"),
                    "detector": _first_header(headers, "DETECTOR", "DETNAM", "DET_ID", "CHANNEL"),
                    "observation_timestamp": _first_header(
                        headers, "DATE-OBS", "DATE_OBS", "DATE-BEG", "MJD-AVG", "MJD-OBS"
                    ),
                    "brightness_unit": _first_header(headers, "BUNIT"),
                    "wcs": self._wcs_metadata(science.header),
                    "hdus": hdu_rows,
                }
        except (OSError, ValueError) as exc:
            raise FitsProcessingError(f"Could not read FITS file {path.name}: {exc}") from exc

    def extract_target(
        self,
        file_path: Path | str,
        ra: float,
        dec: float,
        size: int | tuple[int, int] = 200,
    ) -> dict[str, Any]:
        """Extract a target-centred cutout while leaving the FITS file untouched."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"FITS file not found: {path}")
        ra_value, dec_value = self._validate_coordinate(ra, dec)
        cutout_size = self._validate_size(size)
        try:
            with fits.open(path, mode="readonly", memmap=False) as hdul:
                science_index = self._science_hdu_index(hdul)
                hdu = hdul[science_index]
                image = _as_2d(hdu.data)
                assert image is not None
                wcs = self._celestial_wcs(hdu.header)
                target = SkyCoord(ra_value, dec_value, unit="deg", frame="icrs")
                try:
                    pixel_x, pixel_y = wcs.world_to_pixel(target)
                except Exception as exc:
                    raise InvalidWcsError(f"WCS coordinate conversion failed: {exc}") from exc
                x = float(pixel_x)
                y = float(pixel_y)
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise InvalidWcsError("WCS returned non-finite target pixel coordinates")
                height, width = image.shape
                if not (-0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5):
                    raise TargetOutsideImageError(
                        f"Target is outside image footprint (pixel x={x:.2f}, y={y:.2f})"
                    )
                cutout = Cutout2D(
                    np.asarray(image, dtype=float),
                    position=(x, y),
                    size=cutout_size,
                    wcs=wcs,
                    mode="partial",
                    fill_value=np.nan,
                    copy=True,
                )
                if not np.isfinite(cutout.data).any():
                    raise InvalidImageDataError("Target cutout contains no finite image pixels")
                metadata = self.inspect_fits(path)
                metadata.update(
                    {
                        "target_ra": ra_value,
                        "target_dec": dec_value,
                        "pixel_x": x,
                        "pixel_y": y,
                        "image_width": int(cutout.data.shape[1]),
                        "image_height": int(cutout.data.shape[0]),
                        "source_image_width": width,
                        "source_image_height": height,
                    }
                )
                return {"data": cutout.data, "wcs": cutout.wcs, "metadata": metadata}
        except (FitsProcessingError, FileNotFoundError):
            raise
        except (OSError, ValueError) as exc:
            raise FitsProcessingError(f"Could not process FITS file {path.name}: {exc}") from exc

    def render_preview(
        self,
        cutout: Mapping[str, Any] | np.ndarray,
        output_path: Path | str,
    ) -> dict[str, Any]:
        """Render a grayscale PNG using finite-pixel 2nd/98th percentiles."""
        data = np.asarray(cutout["data"] if isinstance(cutout, Mapping) else cutout, dtype=float)
        if data.ndim != 2 or not data.size:
            raise InvalidImageDataError("Preview data must be a non-empty two-dimensional array")
        finite = data[np.isfinite(data)]
        if not finite.size:
            raise InvalidImageDataError("Preview data contains no finite pixels")
        low, high = (float(value) for value in np.percentile(finite, (2.0, 98.0)))
        constant = not (math.isfinite(low) and math.isfinite(high)) or high <= low
        if constant:
            low = float(np.min(finite))
            high = float(np.max(finite))
            if high <= low:
                high = low + 1.0

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.imsave(destination, data, cmap="gray", vmin=low, vmax=high, origin="lower")
        except Exception as exc:
            raise FitsProcessingError(f"PNG preview generation failed: {exc}") from exc
        return {
            "preview_path": str(destination.resolve()),
            "scale_low": low,
            "scale_high": high,
            "scale_percentiles": [2.0, 98.0],
            "constant_image": constant,
            "finite_pixel_count": int(finite.size),
        }

    def prepare_observation_previews(
        self,
        observations: Sequence[Mapping[str, Any]],
        ra: float,
        dec: float,
        size: int | tuple[int, int] = 200,
        rendering: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate/cached previews for only the supplied observation records."""
        if len(observations) > 6:
            raise ValueError("Select no more than 6 observations for preview generation")
        results: list[dict[str, Any]] = []
        display = {
            "mode": "greyscale",
            "colormap": "gray",
            "stretch": "linear",
            "show_grid": False,
            "show_target": False,
            **dict(rendering or {}),
        }
        display["colormap"] = "gray" if str(display["mode"]).casefold() == "greyscale" else str(display["colormap"]).casefold()
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        for observation in observations:
            result = self._base_result(observation, ra, dec)
            try:
                file_path = self._record_path(observation)
                result["fits_filename"] = file_path.name
                cache_settings = {
                    **display,
                    "observation_id": result["observation_id"],
                    "fits_filename": file_path.name,
                    "detector": observation.get("detector"),
                    "observation_timestamp": observation.get("observation_timestamp")
                    or observation.get("obs_time_iso"),
                    "wcs_rendering_version": 1,
                }
                signature = self._cache_signature(file_path, ra, dec, size, cache_settings)
                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result["observation_id"] or file_path.stem))[:80]
                preview_path = self.preview_dir / f"{safe_id}_{signature}.png"
                metadata_path = preview_path.with_suffix(".json")
                cached = self._read_cache(preview_path, metadata_path, signature)
                if cached is not None:
                    result.update(cached)
                    result.update(status="success", cached=True)
                    results.append(result)
                    continue

                cutout = self.extract_target(file_path, ra, dec, size)
                if display["show_grid"] or display["show_target"]:
                    render = self.renderer.render_wcs_image(
                        cutout,
                        ra,
                        dec,
                        preview_path,
                        colormap=str(display["colormap"]),
                        stretch=str(display["stretch"]),
                        show_grid=bool(display["show_grid"]),
                        show_target=bool(display["show_target"]),
                    )
                else:
                    render = self.renderer.render_false_color(
                        cutout,
                        preview_path,
                        colormap=str(display["colormap"]),
                        stretch=str(display["stretch"]),
                    )
                metadata = cutout["metadata"]
                requested_detector = observation.get("detector")
                fits_detector = metadata.get("detector")
                if (
                    requested_detector not in (None, "")
                    and fits_detector not in (None, "")
                    and str(requested_detector).casefold() != str(fits_detector).casefold()
                ):
                    raise FitsProcessingError(
                        f"Resolved FITS detector {fits_detector} does not match selected detector "
                        f"{requested_detector}"
                    )
                wavelength = self.wavelength_at_target(
                    file_path,
                    metadata["pixel_x"],
                    metadata["pixel_y"],
                    observation,
                )
                result.update(
                    status="success",
                    cached=False,
                    preview_path=render["preview_path"],
                    pixel_x=metadata["pixel_x"],
                    pixel_y=metadata["pixel_y"],
                    image_width=metadata["image_width"],
                    image_height=metadata["image_height"],
                    source_image_width=metadata["source_image_width"],
                    source_image_height=metadata["source_image_height"],
                    brightness_unit=metadata["brightness_unit"],
                    science_hdu_index=metadata["science_hdu_index"],
                    dtype=metadata["dtype"],
                    fits_detector=fits_detector,
                    fits_observation_timestamp=metadata.get("observation_timestamp"),
                    wavelength_um=wavelength["wavelength_um"],
                    wavelength_source=wavelength["source"],
                    rendering=display,
                    scaling={key: value for key, value in render.items() if key != "preview_path"},
                )
                self._write_cache(metadata_path, signature, result)
            except FileNotFoundError as exc:
                result.update(status="missing_file", error=str(exc))
            except TargetOutsideImageError as exc:
                result.update(status="target_outside", error=str(exc))
            except InvalidWcsError as exc:
                result.update(status="invalid_wcs", error=str(exc))
            except InvalidImageDataError as exc:
                result.update(status="invalid_data", error=str(exc))
            except (FitsProcessingError, ImageRenderingError) as exc:
                result.update(status="processing_error", error=str(exc))
            except Exception as exc:
                LOG.exception("Unexpected preview failure for %s", result["observation_id"])
                result.update(status="processing_error", error=f"Unexpected preview failure: {exc}")
            results.append(result)
        return results

    def create_rgb_composite(
        self,
        observations: Sequence[Mapping[str, Any]],
        ra: float,
        dec: float,
        size: int | tuple[int, int] = 200,
        rendering: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract, align and render exactly three original FITS observations."""
        if len(observations) != 3:
            raise ValueError("RGB composite generation requires exactly three observations")
        settings = {
            "stretch": "asinh",
            "show_grid": True,
            "show_target": True,
            **dict(rendering or {}),
        }
        channel_inputs: list[dict[str, Any]] = []
        paths: list[Path] = []
        for observation in observations:
            path = self._record_path(observation)
            paths.append(path)
            cutout = self.extract_target(path, ra, dec, size)
            metadata = cutout["metadata"]
            wavelength = self.wavelength_at_target(path, metadata["pixel_x"], metadata["pixel_y"], observation)
            channel_inputs.append(
                {
                    "data": cutout["data"],
                    "wcs": cutout["wcs"],
                    "observation_id": observation.get("observation_id") or observation.get("obs_id"),
                    "detector": observation.get("detector"),
                    "band": observation.get("band"),
                    "wavelength_um": wavelength["wavelength_um"],
                    "wavelength_source": wavelength["source"],
                    "fits_filename": path.name,
                }
            )
        cache_settings = {
            **settings,
            "channels": [
                {
                    "observation_id": entry["observation_id"],
                    "wavelength_um": entry["wavelength_um"],
                    "wavelength_source": entry["wavelength_source"],
                }
                for entry in channel_inputs
            ],
        }
        signature = self._composite_signature(paths, ra, dec, size, cache_settings)
        destination = self.preview_dir / f"rgb_{signature}.png"
        metadata_path = destination.with_suffix(".json")
        cached = self._read_cache(destination, metadata_path, signature)
        if cached is not None:
            cached.update(status="success", cached=True)
            return cached
        result = self.renderer.create_rgb_composite(
            channel_inputs,
            ra,
            dec,
            destination,
            show_grid=bool(settings["show_grid"]),
            show_target=bool(settings["show_target"]),
            stretch=str(settings["stretch"]),
        )
        result.update(cached=False, fits_filenames=[path.name for path in paths])
        self._write_cache(metadata_path, signature, result)
        return result

    def wavelength_at_target(
        self,
        file_path: Path | str,
        pixel_x: float,
        pixel_y: float,
        observation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a target wavelength only when a defensible source is available."""
        path = Path(file_path)
        try:
            with fits.open(path, mode="readonly", memmap=False) as hdul:
                if "IMAGE" in hdul and "WCS-WAVE" in hdul:
                    spectral_header = hdul["IMAGE"].header.copy()
                    for key in list(spectral_header):
                        if key in {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"} or re.fullmatch(
                            r"(?:A|B|AP|BP)_\d+_\d+", key
                        ):
                            del spectral_header[key]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FITSFixedWarning)
                            spectral_wcs = WCS(spectral_header, fobj=hdul, key="W")
                        spectral_wcs.sip = None
                        wavelength, _bandwidth = spectral_wcs.pixel_to_world(
                            float(pixel_x), float(pixel_y)
                        )
                        value = float(wavelength.to_value(u.um))
                        if math.isfinite(value) and value > 0:
                            return {
                                "wavelength_um": value,
                                "source": "FITS alternate WCS-WAVE at target",
                            }
                    except Exception:
                        pass
                for hdu in hdul:
                    name = str(hdu.header.get("EXTNAME", "")).upper()
                    if not any(token in name for token in ("WCS-WAVE", "WAVE", "LAMBDA")):
                        continue
                    data = hdu.data
                    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
                        array = np.squeeze(data)
                        if array.ndim == 2:
                            x, y = int(round(pixel_x)), int(round(pixel_y))
                            if 0 <= y < array.shape[0] and 0 <= x < array.shape[1]:
                                converted = self._wavelength_to_um(array[y, x], hdu.header.get("BUNIT"))
                                if converted is not None:
                                    return {"wavelength_um": converted, "source": f"FITS {name} image at target"}
                    names = getattr(data, "names", None)
                    if names:
                        for column_name in names:
                            if not any(token in str(column_name).upper() for token in ("WAVE", "LAMBDA")):
                                continue
                            values = np.asarray(data[column_name]).squeeze()
                            if values.size == 1:
                                column = hdu.columns[column_name]
                                converted = self._wavelength_to_um(values.item(), column.unit)
                                if converted is not None:
                                    return {"wavelength_um": converted, "source": f"FITS {name}.{column_name}"}
        except (OSError, ValueError, TypeError):
            pass

        record = observation or {}
        lower = record.get("wavelength_min_um")
        upper = record.get("wavelength_max_um")
        try:
            if lower is not None and upper is not None:
                value = (float(lower) + float(upper)) / 2.0
                if math.isfinite(value):
                    return {"wavelength_um": value, "source": "IRSA band midpoint"}
        except (TypeError, ValueError):
            pass
        return {"wavelength_um": None, "source": None}

    @staticmethod
    def _wavelength_to_um(value: Any, unit: Any) -> float | None:
        if unit in (None, ""):
            return None
        try:
            result = (float(value) * u.Unit(str(unit))).to_value(u.um)
            return float(result) if math.isfinite(result) and result > 0 else None
        except (TypeError, ValueError, u.UnitConversionError):
            return None

    @staticmethod
    def _base_result(observation: Mapping[str, Any], ra: float, dec: float) -> dict[str, Any]:
        return {
            "observation_id": observation.get("observation_id") or observation.get("obs_id"),
            "detector": observation.get("detector") or observation.get("band"),
            "band": observation.get("band"),
            "observation_date": observation.get("observation_timestamp") or observation.get("obs_time_iso"),
            "status": "pending",
            "preview_path": None,
            "target_ra": float(ra),
            "target_dec": float(dec),
            "pixel_x": None,
            "pixel_y": None,
            "image_width": None,
            "image_height": None,
            "brightness_unit": None,
            "error": None,
        }

    @staticmethod
    def _record_path(observation: Mapping[str, Any]) -> Path:
        if observation.get("download_error"):
            raise FileNotFoundError(f"FITS download failed: {observation['download_error']}")
        keys = ("file_path", "fits_path", "local_path", "download_path", "filename")
        for container in (observation, observation.get("raw_metadata", {})):
            if not isinstance(container, Mapping):
                continue
            for key in keys:
                value = container.get(key)
                if value:
                    path = Path(str(value)).expanduser()
                    if path.is_file():
                        return path
                    if path.suffix.lower() in {".fits", ".fit", ".fts"}:
                        raise FileNotFoundError(f"Downloaded FITS file is missing: {path}")
        raise FileNotFoundError("Observation has no downloaded FITS file path")

    @staticmethod
    def _cache_signature(
        path: Path,
        ra: float,
        dec: float,
        size: Any,
        rendering: Mapping[str, Any] | None = None,
    ) -> str:
        stat = path.stat()
        settings = json.dumps(dict(rendering or {}), sort_keys=True, default=str)
        token = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{float(ra):.10f}|{float(dec):.10f}|{size}|{settings}"
        return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _composite_signature(
        cls,
        paths: Sequence[Path],
        ra: float,
        dec: float,
        size: Any,
        rendering: Mapping[str, Any],
    ) -> str:
        tokens = [cls._cache_signature(path, ra, dec, size, rendering) for path in paths]
        return hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _read_cache(preview: Path, metadata: Path, signature: str) -> dict[str, Any] | None:
        if not preview.is_file() or not metadata.is_file():
            return None
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            if payload.pop("_cache_signature", None) != signature:
                return None
            return payload
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _write_cache(path: Path, signature: str, result: Mapping[str, Any]) -> None:
        payload = dict(result)
        payload["_cache_signature"] = signature
        try:
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            LOG.warning("Could not write preview cache metadata %s: %s", path, exc)

    @staticmethod
    def _validate_coordinate(ra: float, dec: float) -> tuple[float, float]:
        try:
            values = float(ra), float(dec)
        except (TypeError, ValueError) as exc:
            raise ValueError("RA and Dec must be finite decimal degrees") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("RA and Dec must be finite decimal degrees")
        if not 0.0 <= values[0] < 360.0 or not -90.0 <= values[1] <= 90.0:
            raise ValueError("RA must be in [0, 360) and Dec in [-90, 90] degrees")
        return values

    @staticmethod
    def _validate_size(size: int | tuple[int, int]) -> tuple[int, int]:
        values = (size, size) if isinstance(size, int) else size
        if len(values) != 2 or any(isinstance(value, bool) or int(value) <= 0 for value in values):
            raise ValueError("Cutout size must contain one or two positive pixel counts")
        return int(values[0]), int(values[1])

    @staticmethod
    def _has_celestial_wcs(header: fits.Header) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                return bool(WCS(header).has_celestial)
        except Exception:
            return False

    @staticmethod
    def _celestial_wcs(header: fits.Header) -> WCS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                wcs = WCS(header)
            if not wcs.has_celestial:
                raise InvalidWcsError("Science image has no celestial WCS")
            celestial = wcs.celestial
            if celestial.pixel_n_dim != 2 or celestial.world_n_dim != 2:
                raise InvalidWcsError("Science image WCS is not two-dimensional")
            return celestial
        except InvalidWcsError:
            raise
        except Exception as exc:
            raise InvalidWcsError(f"Science image WCS is invalid: {exc}") from exc

    @classmethod
    def _wcs_metadata(cls, header: fits.Header) -> dict[str, Any]:
        keys = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2")
        return {
            "has_celestial": cls._has_celestial_wcs(header),
            "header_values": {key: _plain(header.get(key)) for key in keys if key in header},
        }


def inspect_fits(path: Path | str, plot_dir: Path | str | None = None) -> dict[str, Any]:
    """Compatibility wrapper around :class:`SpherexFitsProcessor`."""
    processor = SpherexFitsProcessor(plot_dir or "cache/previews")
    metadata = processor.inspect_fits(path)
    if plot_dir is not None:
        try:
            with fits.open(path, memmap=False) as hdul:
                index = metadata["science_hdu_index"]
                data = _as_2d(hdul[index].data)
                if data is not None and np.isfinite(data).any():
                    processor.render_preview(data, Path(plot_dir) / f"{Path(path).stem}_hdu{index}.png")
        except FitsProcessingError as exc:
            LOG.warning("Could not plot %s: %s", path, exc)
    return metadata


def inspect_directory(raw_dir: Path, plot_dir: Path) -> pd.DataFrame:
    """Inventory local FITS files while retaining the existing pipeline API."""
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.fits")):
        try:
            metadata = inspect_fits(path, plot_dir)
            for hdu in metadata["hdus"]:
                rows.append(
                    {
                        "filename": metadata["filename"],
                        "hdu_index": hdu["index"],
                        "extname": hdu["name"],
                        "classification": hdu["classification"],
                        "shape": str(hdu["shape"]),
                        "dtype": hdu["dtype"],
                        "finite_fraction": hdu["finite_fraction"],
                        "bunit": hdu["brightness_unit"],
                        "has_celestial_wcs": hdu["has_celestial_wcs"],
                    }
                )
        except Exception as exc:
            rows.append(
                {"filename": path.name, "hdu_index": None, "extname": None, "classification": "invalid", "error": str(exc)}
            )
    return pd.DataFrame(rows)
