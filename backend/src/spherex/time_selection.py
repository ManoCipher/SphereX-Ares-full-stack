"""Automatic choice of observations for a time (blink) comparison.

Works on archive metadata only, before anything is downloaded:

1. keep one detector, so the frames are as comparable as metadata allows
   (the exact wavelength at the target is only known after download; the blink
   window warns when it differs);
2. group the frames into epochs: SPHEREx sees the same sky in short bursts
   (several exposures within minutes or hours) about every six months;
3. take the best frame of every epoch first (target closest to the frame
   centre, so it is far from the edges), then second-best ones, and so on,
   until ``count`` frames are chosen or none are left.

If fewer frames exist than requested, all usable ones are returned.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

DEFAULT_FRAME_COUNT = 5
# Frames closer together than this belong to the same survey visit (epoch).
# SPHEREx revisits a sky position about every six months and one visit can be
# spread over several weeks, so anything under a month counts as one epoch.
EPOCH_GAP_DAYS = 30.0
# Auto selection should see every frame, not just the first page of results.
AUTO_SEARCH_MAX_RESULTS = 1000
_MJD_UNIX_EPOCH = 40587.0  # MJD of 1970-01-01


def observation_mjd(observation: Mapping[str, Any]) -> float | None:
    """Observation time as MJD, from the start MJD or the ISO timestamp."""
    value = observation.get("observation_start_mjd")
    try:
        if value is not None and math.isfinite(float(value)):
            return float(value)
    except (TypeError, ValueError):
        pass
    text = observation.get("observation_timestamp")
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp() / 86400.0 + _MJD_UNIX_EPOCH


def _offset_deg(observation: Mapping[str, Any], ra: float, dec: float) -> float:
    """Angular distance from the target to the frame centre (inf if unknown)."""
    try:
        frame_ra = float(observation["ra"])
        frame_dec = float(observation["dec"])
    except (KeyError, TypeError, ValueError):
        return math.inf
    ra1, dec1, ra2, dec2 = map(math.radians, (ra, dec, frame_ra, frame_dec))
    cosine = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def group_epochs(
    observations: Sequence[Mapping[str, Any]], gap_days: float = EPOCH_GAP_DAYS
) -> list[list[Mapping[str, Any]]]:
    """Split time-ordered observations wherever the gap exceeds ``gap_days``."""
    timed = sorted(
        (record for record in observations if observation_mjd(record) is not None),
        key=lambda record: observation_mjd(record),
    )
    epochs: list[list[Mapping[str, Any]]] = []
    previous: float | None = None
    for record in timed:
        moment = observation_mjd(record)
        if previous is None or moment - previous > gap_days:
            epochs.append([])
        epochs[-1].append(record)
        previous = moment
    return epochs


def _product_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("fits_url") or ""),
        str(record.get("observation_id") or ""),
        str(record.get("detector") or ""),
    )


def _best_detector(observations: Sequence[Mapping[str, Any]], gap_days: float) -> str | None:
    """Detector with the most epochs; ties go to more frames, then the lower number."""
    by_detector: dict[str, list[Mapping[str, Any]]] = {}
    for record in observations:
        detector = record.get("detector")
        if detector not in (None, "") and observation_mjd(record) is not None:
            by_detector.setdefault(str(detector), []).append(record)
    if not by_detector:
        return None
    return min(
        by_detector,
        key=lambda name: (-len(group_epochs(by_detector[name], gap_days)), -len(by_detector[name]), name),
    )


def select_time_series(
    observations: Sequence[Mapping[str, Any]],
    ra: float,
    dec: float,
    *,
    count: int = DEFAULT_FRAME_COUNT,
    detector: str | None = None,
    gap_days: float = EPOCH_GAP_DAYS,
) -> dict[str, Any]:
    """Pick up to ``count`` frames of one detector, spread over as many epochs as possible.

    Returns ``{"observations": [...time ordered...], "detector", "epoch_count",
    "available", "note"}``. ``observations`` holds the original records.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    chosen_detector = str(detector) if detector not in (None, "") else _best_detector(observations, gap_days)
    if chosen_detector is None:
        return {"observations": [], "detector": None, "epoch_count": 0, "available": 0,
                "note": "No observations with both a detector and an observation time."}

    seen: set[tuple[str, str, str]] = set()
    pool: list[Mapping[str, Any]] = []
    for record in observations:
        key = _product_key(record)
        if str(record.get("detector")) == chosen_detector and key not in seen:
            seen.add(key)
            pool.append(record)
    epochs = group_epochs(pool, gap_days)
    ranked = [sorted(epoch, key=lambda record: _offset_deg(record, ra, dec)) for epoch in epochs]

    picked: list[Mapping[str, Any]] = []
    depth = 0
    while len(picked) < count and any(depth < len(epoch) for epoch in ranked):
        for epoch in ranked:
            if depth < len(epoch) and len(picked) < count:
                picked.append(epoch[depth])
        depth += 1
    picked.sort(key=lambda record: observation_mjd(record))

    available = sum(len(epoch) for epoch in epochs)
    epoch_of = {id(record): index for index, epoch in enumerate(epochs) for record in epoch}
    used_epochs = len({epoch_of[id(record)] for record in picked})
    if len(picked) < count:
        note = f"Only {len(picked)} frame(s) available for D{chosen_detector}; using all of them."
    else:
        note = f"Picked {len(picked)} of {available} D{chosen_detector} frames."
    note += f" They cover {used_epochs} epoch(s) (survey visits more than {gap_days:g} days apart)."
    if used_epochs < 2:
        note += " With a single epoch the frames are minutes apart, so little or no motion is expected."
    return {
        "observations": picked,
        "detector": chosen_detector,
        "epoch_count": len(epochs),
        "available": available,
        "note": note,
    }
