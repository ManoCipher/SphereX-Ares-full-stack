"""Observation selection without downloading or processing science products.

The selector deliberately has two phases.  Archive metadata is used to remove
duplicates and impossible footprints, then the small shortlist is checked
against already-local FITS files (or an injected lightweight remote metadata
validator).  A missing local file never causes a full FITS download.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from .fits_processor import SpherexFitsProcessor
from .time_selection import EPOCH_GAP_DAYS, group_epochs, observation_mjd

MAX_SELECTION = 6
WAVELENGTH_TOLERANCE_FRACTION = 0.05


class SelectionMode(str, Enum):
    SPECTRAL = "Spectral Comparison"
    TIME = "Time Comparison"
    MANUAL = "Manual Selection"


@dataclass(frozen=True)
class SelectionResult:
    observations: tuple[Mapping[str, Any], ...]
    mode: SelectionMode
    explanation: str


@dataclass
class _Candidate:
    record: Mapping[str, Any]
    detector: str
    mjd: float | None
    wavelength_um: float | None
    quality: float
    edge_margin: float


Validator = Callable[[Mapping[str, Any], float, float], Mapping[str, Any] | None]
PathResolver = Callable[[Mapping[str, Any]], Path | str | None]


def _product_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("fits_url") or ""),
        str(record.get("observation_id") or record.get("obs_id") or ""),
        str(record.get("detector") or ""),
    )


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _wavelength(record: Mapping[str, Any]) -> float | None:
    for key in ("target_wavelength_um", "wavelength_um", "central_wavelength_um"):
        value = _number(record.get(key))
        if value is not None and value > 0:
            return value
    lower = _number(record.get("wavelength_min_um"))
    upper = _number(record.get("wavelength_max_um"))
    if lower is not None and upper is not None and lower > 0 and upper > 0:
        return (lower + upper) / 2.0
    return None


def _angular_distance(ra: float, dec: float, other_ra: float, other_dec: float) -> float:
    first = SkyCoord(ra, dec, unit="deg", frame="icrs")
    second = SkyCoord(other_ra, other_dec, unit="deg", frame="icrs")
    return float(first.separation(second).deg)


def _unwrap_ra(value: float, around: float) -> float:
    return around + ((value - around + 180.0) % 360.0 - 180.0)


def _point_in_polygon(ra: float, dec: float, vertices: Sequence[tuple[float, float]]) -> bool:
    """Adequate local tangent-plane test for the small SIA image polygons."""
    points = [(_unwrap_ra(x, ra) * math.cos(math.radians(dec)), y) for x, y in vertices]
    x = ra * math.cos(math.radians(dec))
    inside = False
    previous = len(points) - 1
    for current, (xi, yi) in enumerate(points):
        xj, yj = points[previous]
        if (yi > dec) != (yj > dec):
            boundary = (xj - xi) * (dec - yi) / (yj - yi) + xi
            if x < boundary:
                inside = not inside
        previous = current
    return inside


def _metadata_coverage(record: Mapping[str, Any], ra: float, dec: float) -> bool | None:
    explicit = record.get("target_covered")
    if explicit is not None:
        return bool(explicit)
    region = record.get("spatial_coverage") or record.get("s_region")
    if isinstance(region, str):
        numbers = [_number(value) for value in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", region)]
        values = [value for value in numbers if value is not None]
        upper = region.upper()
        if "CIRCLE" in upper and len(values) >= 3:
            return _angular_distance(ra, dec, values[-3], values[-2]) <= values[-1]
        if "POLYGON" in upper and len(values) >= 6 and len(values) % 2 == 0:
            return _point_in_polygon(ra, dec, list(zip(values[0::2], values[1::2], strict=True)))
    centre_ra = _number(record.get("ra"))
    centre_dec = _number(record.get("dec"))
    fov = _number(record.get("field_of_view_deg"))
    if centre_ra is not None and centre_dec is not None and fov is not None and fov > 0:
        # SIA s_fov is the diameter of a bounding circle, not guaranteed image coverage.
        if _angular_distance(ra, dec, centre_ra, centre_dec) > fov / 2.0:
            return False
    return None


def _metadata_quality(record: Mapping[str, Any]) -> tuple[bool, float]:
    if record.get("quality_ok") is False or record.get("usable") is False:
        return False, 0.0
    raw = record.get("raw_metadata")
    sources = (record, raw if isinstance(raw, Mapping) else {})
    for source in sources:
        for key in ("quality_flag", "data_quality", "qual_flag"):
            value = source.get(key)
            if isinstance(value, str) and value.strip().casefold() in {"bad", "fail", "failed", "reject"}:
                return False, 0.0
            numeric = _number(value)
            if numeric is not None and numeric < 0:
                return False, 0.0
    score = _number(record.get("quality_score"))
    return True, max(0.0, min(1.0, score if score is not None else 0.5))


class ObservationSelectionService:
    """Select comparison records while remaining independent of the UI/download path."""

    def __init__(
        self,
        *,
        path_resolver: PathResolver | None = None,
        remote_validator: Validator | None = None,
        max_selection: int = MAX_SELECTION,
    ) -> None:
        if not 1 <= max_selection <= MAX_SELECTION:
            raise ValueError(f"max_selection must be between 1 and {MAX_SELECTION}")
        self.path_resolver = path_resolver
        self.remote_validator = remote_validator
        self.max_selection = max_selection
        self._fits_processor = SpherexFitsProcessor()

    def select(
        self,
        observations: Sequence[Mapping[str, Any]],
        ra: float,
        dec: float,
        mode: SelectionMode | str,
        *,
        manual_selection: Sequence[Mapping[str, Any]] | None = None,
    ) -> SelectionResult:
        selected_mode = SelectionMode(mode)
        if selected_mode is SelectionMode.MANUAL:
            chosen = tuple(manual_selection or ())
            if len(chosen) > self.max_selection:
                raise ValueError(f"Manual selection is limited to {self.max_selection} observations")
            return SelectionResult(chosen, selected_mode, f"Manual selection preserved ({len(chosen)} observation(s)).")

        ra_value, dec_value = float(ra), float(dec)
        candidates = self._validated_shortlist(observations, ra_value, dec_value)
        if selected_mode is SelectionMode.SPECTRAL:
            chosen = self._select_spectral(candidates)
            detectors = ", ".join(f"D{item.detector}" for item in chosen)
            explanation = (
                f"Selected {len(chosen)} observation(s) across {len({item.detector for item in chosen})} detector(s)"
                + (f" ({detectors})" if detectors else "")
                + "; prioritised close times, target coverage, wavelength metadata and quality."
            )
        else:
            chosen = self._select_time(candidates)
            detector = chosen[0].detector if chosen else None
            visits = len(group_epochs([item.record for item in chosen], EPOCH_GAP_DAYS))
            explanation = (
                f"Selected {len(chosen)} comparable observation(s)"
                + (f" from D{detector}" if detector else "")
                + f" across {visits} survey visit(s); prioritised time span, target coverage, wavelength and quality."
            )
        return SelectionResult(tuple(item.record for item in chosen), selected_mode, explanation)

    def _validated_shortlist(
        self, observations: Sequence[Mapping[str, Any]], ra: float, dec: float
    ) -> list[_Candidate]:
        seen: set[tuple[str, str, str]] = set()
        shortlisted: list[_Candidate] = []
        for record in observations:
            detector = str(record.get("detector") or "").removeprefix("D").removeprefix("d")
            key = _product_key(record)
            if not detector or key in seen:
                continue
            seen.add(key)
            coverage = _metadata_coverage(record, ra, dec)
            quality_ok, quality = _metadata_quality(record)
            if coverage is False or not quality_ok:
                continue
            validation = self._validate_candidate(record, ra, dec)
            if validation is not None:
                if validation.get("covered") is False or validation.get("quality_ok") is False:
                    continue
                coverage = bool(validation.get("covered", coverage))
                quality = float(validation.get("quality_score", quality))
                wavelength = _number(validation.get("wavelength_um")) or _wavelength(record)
                edge_margin = float(validation.get("edge_margin", 0.0))
            else:
                wavelength = _wavelength(record)
                edge_margin = 0.0
            # Detailed archive footprints count as stage-two coverage when no
            # local/remote validator is available. Unknown centre-only geometry does not.
            if coverage is not True:
                continue
            shortlisted.append(
                _Candidate(record, detector, observation_mjd(record), wavelength, quality, edge_margin)
            )
        return shortlisted

    def _validate_candidate(
        self, record: Mapping[str, Any], ra: float, dec: float
    ) -> Mapping[str, Any] | None:
        path = self.path_resolver(record) if self.path_resolver else None
        if path and Path(path).is_file():
            return self._validate_local_fits(Path(path), record, ra, dec)
        if self.remote_validator:
            return self.remote_validator(record, ra, dec)
        return None

    def _validate_local_fits(
        self, path: Path, record: Mapping[str, Any], ra: float, dec: float
    ) -> Mapping[str, Any]:
        try:
            with fits.open(path, mode="readonly", memmap=False) as hdul:
                choices: list[tuple[Any, np.ndarray, WCS]] = []
                for hdu in hdul:
                    data = np.squeeze(hdu.data) if isinstance(hdu.data, np.ndarray) else None
                    if data is None or data.ndim != 2 or not np.issubdtype(data.dtype, np.number):
                        continue
                    try:
                        wcs = WCS(hdu.header, fobj=hdul).celestial
                        if wcs.pixel_n_dim != 2:
                            continue
                    except Exception:
                        continue
                    choices.append((hdu, data, wcs))
                if not choices:
                    return {"covered": False, "quality_ok": False}
                hdu, data, wcs = choices[0]
                x, y = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg", frame="icrs"))
                height, width = data.shape
                covered = bool(
                    math.isfinite(float(x)) and math.isfinite(float(y))
                    and -0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5
                )
                if not covered:
                    return {"covered": False}
                margin = min(float(x) + 0.5, float(y) + 0.5, width - 0.5 - float(x), height - 0.5 - float(y))
                edge_margin = max(0.0, margin / max(1.0, min(width, height) / 2.0))
                radius = 5
                xi, yi = int(round(float(x))), int(round(float(y)))
                sample = data[max(0, yi - radius): yi + radius + 1, max(0, xi - radius): xi + radius + 1]
                finite_fraction = float(np.isfinite(sample).mean()) if sample.size else 0.0
                header_quality = str(hdu.header.get("QUALITY", hdu.header.get("DATAQUAL", ""))).casefold()
                quality_ok = finite_fraction >= 0.5 and header_quality not in {"bad", "fail", "failed", "reject"}
            wavelength = self._fits_processor.wavelength_at_target(path, float(x), float(y), record)
            return {
                "covered": True,
                "quality_ok": quality_ok,
                "quality_score": finite_fraction,
                "edge_margin": edge_margin,
                "wavelength_um": wavelength.get("wavelength_um"),
            }
        except Exception:
            return {"covered": False, "quality_ok": False}

    @staticmethod
    def _time_spread(items: Sequence[_Candidate]) -> float:
        times = [item.mjd for item in items if item.mjd is not None]
        return max(times) - min(times) if len(times) >= 2 else 0.0

    def _select_spectral(self, candidates: Sequence[_Candidate]) -> list[_Candidate]:
        timed = [item for item in candidates if item.mjd is not None]
        pool = timed or list(candidates)
        if not pool:
            return []
        best: list[_Candidate] = []
        best_key: tuple[Any, ...] | None = None
        anchors = [item.mjd for item in pool if item.mjd is not None] or [None]
        for anchor in anchors:
            per_detector: list[_Candidate] = []
            for detector in sorted({item.detector for item in pool}, key=lambda value: (not value.isdigit(), value)):
                choices = [item for item in pool if item.detector == detector]
                per_detector.append(min(choices, key=lambda item: (
                    abs((item.mjd or anchor or 0.0) - (anchor or item.mjd or 0.0)),
                    item.wavelength_um is None, -item.quality, -item.edge_margin,
                )))
            choice = per_detector[: self.max_selection]
            key = (-len(choice), self._time_spread(choice), -sum(item.quality + item.edge_margin for item in choice))
            if best_key is None or key < best_key:
                best_key, best = key, choice
        return sorted(best, key=lambda item: (int(item.detector) if item.detector.isdigit() else 99, item.detector))

    @staticmethod
    def _wavelength_groups(candidates: Sequence[_Candidate]) -> list[list[_Candidate]]:
        known = sorted((item for item in candidates if item.wavelength_um), key=lambda item: item.wavelength_um or 0.0)
        unknown = [item for item in candidates if item.wavelength_um is None]
        groups: list[list[_Candidate]] = []
        for item in known:
            placed = False
            for group in groups:
                centre = float(np.median([member.wavelength_um for member in group]))
                if abs((item.wavelength_um or centre) - centre) / centre <= WAVELENGTH_TOLERANCE_FRACTION:
                    group.append(item)
                    placed = True
                    break
            if not placed:
                groups.append([item])
        if not groups and unknown:
            return [unknown]
        # Unknown wavelength is a weaker fallback for every detector, but never
        # displaces a larger set with measured comparable wavelengths.
        if unknown:
            groups.append(unknown)
        return groups

    def _select_time(self, candidates: Sequence[_Candidate]) -> list[_Candidate]:
        options: list[list[_Candidate]] = []
        for detector in {item.detector for item in candidates}:
            detector_items = [item for item in candidates if item.detector == detector and item.mjd is not None]
            options.extend(self._wavelength_groups(detector_items))
        if not options:
            return []
        pool = min(options, key=lambda group: (
            -len(group_epochs([item.record for item in group], EPOCH_GAP_DAYS)),
            -self._time_spread(group), -len(group),
            -sum(item.quality + item.edge_margin for item in group), group[0].detector,
        ))
        epochs = group_epochs([item.record for item in pool], EPOCH_GAP_DAYS)
        by_id = {id(item.record): item for item in pool}
        ranked = [sorted((by_id[id(record)] for record in epoch), key=lambda item: (-item.quality, -item.edge_margin)) for epoch in epochs]
        picked: list[_Candidate] = []
        # First cover distinct visits; endpoints are retained when visits exceed the limit.
        visit_indices = list(range(len(ranked)))
        if len(visit_indices) > self.max_selection:
            if self.max_selection == 1:
                visit_indices = [len(ranked) // 2]
            else:
                visit_indices = [
                    round(index * (len(ranked) - 1) / (self.max_selection - 1))
                    for index in range(self.max_selection)
                ]
        for index in visit_indices:
            picked.append(ranked[index][0])
        depth = 1
        while len(picked) < self.max_selection and any(depth < len(epoch) for epoch in ranked):
            choices = [epoch[depth] for epoch in ranked if depth < len(epoch)]
            choices.sort(key=lambda item: min(abs((item.mjd or 0) - (chosen.mjd or 0)) for chosen in picked), reverse=True)
            picked.extend(choices[: self.max_selection - len(picked)])
            depth += 1
        return sorted(picked, key=lambda item: item.mjd or math.inf)
