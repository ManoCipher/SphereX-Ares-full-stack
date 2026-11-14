"""Conservative, non-destructive cleaning for SPHEREx scientific images."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy.wcs import WCS


PIPELINE_VERSION = "spherex-cleaner-1"

# SPHEREx Explanatory Supplement v2.1, Table 16.  FITS MP_* header
# definitions take precedence because they describe the product being read.
DOCUMENTED_FLAG_BITS = {
    "TRANSIENT": 0,
    "OVERFLOW": 1,
    "SUR_ERROR": 2,
    "NONFUNC": 6,
    "DICHROIC": 7,
    "MISSING_DATA": 9,
    "HOT": 10,
    "COLD": 11,
    "FULLSAMPLE": 12,
    "PHANMISS": 14,
    "NONLINEAR": 15,
    "PERSIST": 17,
    "OUTLIER": 19,
    "CROSSTALK": 20,
    "SOURCE": 21,
    "GHOST": 22,
    "GHOST_FPA": 23,
    "GHOST_EXT": 24,
    "STREAK": 25,
    "BLOOM": 26,
    "SNOWBALL": 27,
    "HALO": 28,
    "SATELLITE_HALO": 29,
}

SCIENCE_INVALID_FLAGS = {
    "SUR_ERROR",
    "NONFUNC",
    "DICHROIC",
    "MISSING_DATA",
    "HOT",
    "COLD",
    "PHANMISS",
    "NONLINEAR",
    "PERSIST",
    "BLOOM",
    "SNOWBALL",
    "HALO",
    "SATELLITE_HALO",
}
BACKGROUND_EXCLUDE_FLAGS = SCIENCE_INVALID_FLAGS | {
    "TRANSIENT",
    "OVERFLOW",
    "OUTLIER",
    "SOURCE",
    "CROSSTALK",
    "GHOST",
    "GHOST_FPA",
    "GHOST_EXT",
    "STREAK",
}
ARTEFACT_FLAGS = {
    "CROSSTALK",
    "GHOST",
    "GHOST_FPA",
    "GHOST_EXT",
    "STREAK",
    "BLOOM",
    "SNOWBALL",
    "HALO",
    "SATELLITE_HALO",
}
ARTEFACT_HEADER_KEYS = {
    "L2.SATSTRKS_PXS",
    "L2.N_CROSSTALK",
    "L2.N_GHOST",
    "L2.N_GHOST_FPA",
    "L2.N_GHOST_EXT",
    "L2.N_BLOOM",
    "L2.N_SNOWBALL",
    "L2.N_HALO",
    "L2.N_SAT_HALO",
}


def default_cleaning_settings() -> dict[str, bool]:
    return {
        "enabled": True,
        "quality_masking": True,
        "background_correction": True,
        "artefact_inspection": True,
    }


class SpherexImageCleaner:
    """Clean arrays without changing the input arrays or source FITS file."""

    pipeline_version = PIPELINE_VERSION

    def clean(
        self,
        image: np.ndarray,
        flags: np.ndarray | None = None,
        variance: np.ndarray | None = None,
        header: Mapping[str, Any] | None = None,
        zodi: np.ndarray | None = None,
        *,
        settings: Mapping[str, Any] | None = None,
        wcs: WCS | None = None,
    ) -> dict[str, Any]:
        """Return cleaned measurements, separate masks, diagnostics and provenance."""
        original = np.asarray(image)
        if original.ndim != 2 or not original.size:
            raise ValueError("Scientific image must be a non-empty two-dimensional array")
        shape = original.shape
        config = {**default_cleaning_settings(), **dict(settings or {})}
        enabled = bool(config["enabled"])
        quality_on = enabled and bool(config["quality_masking"])
        background_on = enabled and bool(config["background_correction"])
        artefact_on = enabled and bool(config["artefact_inspection"])
        header_map = header or {}

        science = np.array(original, dtype=float, copy=True)
        flag_array = self._matching_array(flags, shape, dtype=np.uint64, fill=0)
        variance_array = self._matching_array(variance, shape, dtype=float, fill=np.nan)
        zodi_array = self._matching_array(zodi, shape, dtype=float, fill=np.nan)
        bit_positions = self._flag_bits(header_map)
        decoded_counts = {
            name: int(np.count_nonzero(flag_array & (np.uint64(1) << np.uint64(bit))))
            for name, bit in bit_positions.items()
        }

        invalid_measurement = ~np.isfinite(science)
        invalid_variance = np.zeros(shape, dtype=bool)
        if variance is not None:
            invalid_variance = ~np.isfinite(variance_array) | (variance_array <= 0)
        flagged_invalid = self._mask_for_names(flag_array, bit_positions, SCIENCE_INVALID_FLAGS)
        quality_mask = invalid_measurement | invalid_variance
        if quality_on:
            quality_mask |= flagged_invalid

        artefact_mask = np.zeros(shape, dtype=bool)
        if artefact_on:
            artefact_mask = self._mask_for_names(flag_array, bit_positions, ARTEFACT_FLAGS)

        background = np.zeros(shape, dtype=float)
        background_exclusion = quality_mask.copy()
        background_exclusion |= self._mask_for_names(
            flag_array, bit_positions, BACKGROUND_EXCLUDE_FLAGS
        )
        source_mask = self._source_aware_mask(science, background_exclusion)
        background_exclusion |= source_mask
        zodi_subtracted = False
        warnings: list[str] = []

        if background_on:
            zodi_already_subtracted = self._header_truth(
                header_map, "ZODISUB", "ZODI_SUB", "ZL_SUB", "ZLSUB"
            )
            if zodi is not None and not zodi_already_subtracted:
                valid_zodi = np.isfinite(zodi_array)
                background[valid_zodi] += zodi_array[valid_zodi]
                zodi_subtracted = bool(valid_zodi.any())
            elif zodi is not None and zodi_already_subtracted:
                warnings.append("ZODI model was not subtracted because the header marks it as already removed.")
            residual_input = science - background
            residual_background = self._large_scale_background(
                residual_input, background_exclusion
            )
            background += residual_background
            science -= background

        # Keep masking distinct from correction: measurements are not interpolated.
        if quality_on:
            science[quality_mask] = np.nan

        artefact_counts = {
            name: decoded_counts.get(name, 0)
            for name in sorted(ARTEFACT_FLAGS)
            if decoded_counts.get(name, 0)
        }
        header_artefact_indicators = self._header_counts(header_map, ARTEFACT_HEADER_KEYS)
        if artefact_counts or header_artefact_indicators:
            warnings.append(
                "Artefact metadata is present; flagged pixels are marked in artefact_mask where available "
                "and were not interpolated."
            )
        unknown_bits = self._unknown_flag_values(flag_array, bit_positions)
        if unknown_bits:
            warnings.append("FLAGS contains undocumented bits; they were retained but not auto-masked.")

        finite_original = np.isfinite(original)
        finite_cleaned = np.isfinite(science)
        statistics = {
            "pixel_count": int(original.size),
            "original_finite_fraction": float(finite_original.mean()),
            "cleaned_finite_fraction": float(finite_cleaned.mean()),
            "quality_mask_fraction": float(quality_mask.mean()),
            "artefact_mask_fraction": float(artefact_mask.mean()),
            "background_source_exclusion_fraction": float(background_exclusion.mean()),
            "background_median": self._finite_median(background),
            "original_median": self._finite_median(original),
            "cleaned_median": self._finite_median(science),
            "decoded_flag_counts": decoded_counts,
            "artefact_flag_counts": artefact_counts,
            "artefact_header_indicators": header_artefact_indicators,
            "unknown_flag_value_or": unknown_bits,
        }
        provenance = {
            "pipeline": self.pipeline_version,
            "flag_definition_source": "FITS MP_* keywords with SPHEREx Table 16 fallback",
            "zodi_action": "subtracted_once" if zodi_subtracted else "not_subtracted",
            "background_method": "source-aware sigma-clipped coarse-grid interpolation",
            "measurement_replacement": "none; quality-masked output pixels are NaN",
        }
        original_wcs = wcs.deepcopy() if wcs is not None else self._wcs_from_header(header_map)
        return {
            "original_image": original,
            "cleaned_image": science,
            "background_corrected": science,
            "quality_mask": quality_mask,
            "artefact_mask": artefact_mask,
            "background": background,
            "background_model": background,
            "statistics": statistics,
            "parameters": config,
            "provenance": provenance,
            "warnings": warnings,
            "wcs": original_wcs,
            "original_wcs": original_wcs,
        }

    @staticmethod
    def _matching_array(
        value: np.ndarray | None,
        shape: tuple[int, int],
        *,
        dtype: Any,
        fill: float | int,
    ) -> np.ndarray:
        if value is None:
            return np.full(shape, fill, dtype=dtype)
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f"Auxiliary array shape {array.shape} does not match image shape {shape}")
        if dtype == np.uint64:
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError("FLAGS must contain integer bit masks")
            return array.astype(np.uint64, copy=False)
        return array.astype(dtype, copy=False)

    @staticmethod
    def _flag_bits(header: Mapping[str, Any]) -> dict[str, int]:
        result = dict(DOCUMENTED_FLAG_BITS)
        for key in header:
            name = str(key).upper()
            if not name.startswith("MP_"):
                continue
            try:
                bit = int(header[key])
            except (TypeError, ValueError):
                continue
            if 0 <= bit < 64:
                result[name[3:]] = bit
        return result

    @staticmethod
    def _mask_for_names(
        flags: np.ndarray, bits: Mapping[str, int], names: set[str]
    ) -> np.ndarray:
        mask = np.zeros(flags.shape, dtype=bool)
        for name in names:
            bit = bits.get(name)
            if bit is not None:
                mask |= (flags & (np.uint64(1) << np.uint64(bit))) != 0
        return mask

    @staticmethod
    def _source_aware_mask(data: np.ndarray, excluded: np.ndarray) -> np.ndarray:
        usable = np.isfinite(data) & ~excluded
        values = data[usable]
        if values.size < 16:
            return np.zeros(data.shape, dtype=bool)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0:
            return np.zeros(data.shape, dtype=bool)
        # Only mask positive source-like excursions; broad negative sky structure remains estimable.
        return usable & (data > median + 4.0 * sigma)

    @classmethod
    def _large_scale_background(cls, data: np.ndarray, excluded: np.ndarray) -> np.ndarray:
        height, width = data.shape
        grid_y = min(8, max(1, height // 16))
        grid_x = min(8, max(1, width // 16))
        y_edges = np.linspace(0, height, grid_y + 1, dtype=int)
        x_edges = np.linspace(0, width, grid_x + 1, dtype=int)
        coarse = np.full((grid_y, grid_x), np.nan, dtype=float)
        for row in range(grid_y):
            for column in range(grid_x):
                block = data[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]]
                block_mask = excluded[
                    y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]
                ]
                values = block[np.isfinite(block) & ~block_mask]
                coarse[row, column] = cls._sigma_clipped_median(values)
        finite = np.isfinite(coarse)
        fallback = cls._sigma_clipped_median(data[np.isfinite(data) & ~excluded])
        if not np.isfinite(fallback):
            return np.zeros(data.shape, dtype=float)
        coarse[~finite] = fallback
        x_centres = (x_edges[:-1] + x_edges[1:] - 1) / 2.0
        y_centres = (y_edges[:-1] + y_edges[1:] - 1) / 2.0
        x_pixels = np.arange(width, dtype=float)
        y_pixels = np.arange(height, dtype=float)
        along_x = np.vstack([np.interp(x_pixels, x_centres, row) for row in coarse])
        return np.vstack(
            [np.interp(y_pixels, y_centres, along_x[:, column]) for column in range(width)]
        ).T

    @staticmethod
    def _sigma_clipped_median(values: np.ndarray) -> float:
        working = np.asarray(values, dtype=float)
        working = working[np.isfinite(working)]
        for _ in range(3):
            if working.size == 0:
                return float("nan")
            median = float(np.median(working))
            mad = float(np.median(np.abs(working - median)))
            sigma = 1.4826 * mad
            if sigma <= 0 or not np.isfinite(sigma):
                return median
            retained = working[np.abs(working - median) <= 3.0 * sigma]
            if retained.size == working.size:
                return median
            working = retained
        return float(np.median(working)) if working.size else float("nan")

    @staticmethod
    def _header_truth(header: Mapping[str, Any], *keys: str) -> bool:
        for key in keys:
            value = header.get(key)
            if isinstance(value, str):
                if value.strip().upper() in {"T", "TRUE", "YES", "Y", "1"}:
                    return True
            elif value is True or value == 1:
                return True
        return False

    @staticmethod
    def _header_counts(header: Mapping[str, Any], keys: set[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for key in keys:
            value = header.get(key)
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count > 0:
                counts[key] = count
        return counts

    @staticmethod
    def _finite_median(array: np.ndarray) -> float | None:
        values = np.asarray(array)[np.isfinite(array)]
        return float(np.median(values)) if values.size else None

    @staticmethod
    def _unknown_flag_values(flags: np.ndarray, bits: Mapping[str, int]) -> int:
        known = np.uint64(0)
        for bit in bits.values():
            known |= np.uint64(1) << np.uint64(bit)
        combined = np.bitwise_or.reduce(flags.ravel(), initial=np.uint64(0))
        return int(combined & ~known)

    @staticmethod
    def _wcs_from_header(header: Mapping[str, Any]) -> WCS | None:
        try:
            wcs = WCS(header)
            return wcs.celestial if wcs.has_celestial else None
        except Exception:
            return None
