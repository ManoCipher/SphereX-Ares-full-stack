"""Backend IRSA archive discovery client."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from astropy.time import Time

from .config import ArchiveConfig
from .targets import Target

LOG = logging.getLogger(__name__)

OBSERVATION_COLUMNS = [
    "obs_id", "obs_time_mjd", "obs_time_iso", "band_min_um", "band_max_um",
    "data_release", "collection", "access_url", "access_format", "access_estsize_kb",
    "target_ra_deg", "target_dec_deg", "target_uncertainty_arcsec", "metadata_json",
]


class ArchiveQueryError(RuntimeError):
    pass


def _plain(value: Any) -> Any:
    if np.ma.is_masked(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _first(row: Mapping[str, Any], *names: str) -> Any:
    lower = {str(key).lower(): _plain(value) for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def parse_archive_rows(rows: Iterable[Mapping[str, Any]], collection: str, target: Target) -> pd.DataFrame:
    parsed: list[dict[str, Any]] = []
    for raw in rows:
        row = {str(key): _plain(value) for key, value in raw.items()}
        t_min = _first(row, "t_min", "mjd_obs")
        try:
            obs_time = Time(float(t_min), format="mjd", scale="utc") if t_min is not None else None
        except (TypeError, ValueError):
            obs_time = None
        propagated = target.at(obs_time) if obs_time is not None else target.coordinate
        em_min = _first(row, "em_min")
        em_max = _first(row, "em_max")
        parsed.append({
            "obs_id": str(_first(row, "obs_id", "obs_publisher_did", "publisher_did", "title") or "unknown"),
            "obs_time_mjd": float(obs_time.mjd) if obs_time is not None else np.nan,
            "obs_time_iso": obs_time.utc.isot if obs_time is not None else None,
            "band_min_um": float(em_min) * 1e6 if em_min is not None else np.nan,
            "band_max_um": float(em_max) * 1e6 if em_max is not None else np.nan,
            "data_release": collection.removeprefix("spherex_").replace("_deep", ""),
            "collection": collection,
            "access_url": _first(row, "access_url"),
            "access_format": _first(row, "access_format"),
            "access_estsize_kb": _first(row, "access_estsize"),
            "target_ra_deg": propagated.ra.deg,
            "target_dec_deg": propagated.dec.deg,
            "target_uncertainty_arcsec": target.uncertainty_arcsec(obs_time) if obs_time is not None else np.nan,
            "metadata_json": json.dumps(row, sort_keys=True, default=str),
        })
    return pd.DataFrame(parsed, columns=OBSERVATION_COLUMNS)


class IrsaSiaClient:
    def __init__(self, config: ArchiveConfig | None = None) -> None:
        self.config = config or ArchiveConfig()

    def _search_collection(self, target: Target, collection: str, epoch: Time) -> pd.DataFrame:
        try:
            from pyvo.dal.sia2 import SIA2Service

            center = target.at(epoch)
            service = SIA2Service(self.config.sia_url, check_baseurl=False)
            results = service.search(
                pos=(center.ra.deg, center.dec.deg, self.config.search_radius_deg),
                collection=collection,
                maxrec=5000,
            )
            table = results.to_table()
            rows = ({name: row[name] for name in table.colnames} for row in table)
            return parse_archive_rows(rows, collection, target)
        except Exception as exc:  # pyVO raises several transport/protocol exception types
            raise ArchiveQueryError(f"IRSA SIA2 query failed for {collection}: {exc}") from exc

    def discover(
        self,
        target: Target,
        collections: tuple[str, ...] | None = None,
        epoch: Time | None = None,
        fallback_on_error: bool = True,
    ) -> pd.DataFrame:
        """Query all selected collections and combine their observation metadata."""
        epoch = epoch or Time.now()
        errors: list[str] = []
        selected_collections = collections or self.config.collections
        frames: list[pd.DataFrame] = []
        for collection in selected_collections:
            try:
                frame = self._search_collection(target, collection, epoch)
            except ArchiveQueryError as exc:
                errors.append(str(exc))
                LOG.warning("%s", exc)
                if not fallback_on_error:
                    raise
                continue
            if not frame.empty:
                frames.append(frame)
            else:
                LOG.info("No matching records in %s", collection)
        if errors and len(errors) == len(selected_collections):
            raise ArchiveQueryError("; ".join(errors))
        if frames:
            return (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=["access_url", "obs_id"])
                .reset_index(drop=True)
            )
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
