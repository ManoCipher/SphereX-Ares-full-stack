"""Backend metadata client for the IRSA SPHEREx SIA2 archive."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from io import BytesIO
from typing import Any, TypeAlias, TypedDict
from urllib.parse import urlparse

import numpy as np
import requests
from astropy.io.votable import parse as parse_votable
from astropy.io.votable.exceptions import W48
from astropy.time import Time


IRSA_SIA2_URL = "https://irsa.ipac.caltech.edu/SIA"
SPHEREX_COLLECTIONS = (
    "spherex_dr1",
    "spherex_dr1_deep",
    "spherex_qr3",
    "spherex_qr3_deep",
)

DateInput: TypeAlias = str | date | datetime


class SPHERExObservation(TypedDict):
    """Normalized fields from one SPHEREx spectral-image record."""

    observation_id: str | None
    observation_timestamp: str | None
    observation_start_mjd: float | None
    observation_end_mjd: float | None
    ra: float | None
    dec: float | None
    spatial_coverage: str | None
    field_of_view_deg: float | None
    detector: str | None
    band: str | None
    wavelength_min_um: float | None
    wavelength_max_um: float | None
    fits_url: str | None
    exposure_time_s: float | None
    file_size_bytes: int | None
    file_size_kb: float | None
    collection: str | None
    access_format: str | None
    raw_metadata: dict[str, Any]


class SPHERExArchiveError(RuntimeError):
    """Raised when IRSA cannot be reached or returns an invalid SIA2 response."""


def _plain(value: Any) -> Any:
    """Convert Astropy/Numpy values into JSON-friendly Python values."""
    if value is None or np.ma.is_masked(value):
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


def _first(row: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): _plain(value) for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class SPHERExClient:
    """Search IRSA for SPHEREx FITS observations without downloading FITS data.

    The client queries all currently documented DR1 and QR3 wide/deep SIA2
    collections by default.  A custom ``requests.Session`` can be injected for
    tests or application-specific transport configuration.
    """

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        endpoint: str = IRSA_SIA2_URL,
        collections: Sequence[str] = SPHEREX_COLLECTIONS,
        session: requests.Session | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds")
        if not collections:
            raise ValueError("at least one SPHEREx collection is required")
        self.timeout = float(timeout)
        self.endpoint = endpoint
        self.collections = tuple(collections)
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent", "SPHEREx-Sky-Explorer/1.0 (metadata-only SIA2 client)"
        )

    def search(
        self,
        ra: float,
        dec: float,
        radius: float = 0.05,
        max_results: int = 50,
        start_date: DateInput | None = None,
        end_date: DateInput | None = None,
    ) -> list[SPHERExObservation]:
        """Return SPHEREx observations whose footprints overlap a sky circle.

        Coordinates and radius are decimal degrees.  IRSA performs the spherical
        intersection, so circles crossing 0/360 degrees are handled without a
        rectangular RA boundary special case. Dates may be ISO strings,
        ``date`` instances, or ``datetime`` instances.
        """
        ra_value, dec_value, radius_value = self._validate_position(ra, dec, radius)
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results <= 0:
            raise ValueError("max_results must be a positive integer")

        start_mjd = self._date_to_mjd(start_date, "start_date")
        end_mjd = self._date_to_mjd(end_date, "end_date")
        if start_mjd is not None and end_mjd is not None and start_mjd > end_mjd:
            raise ValueError("start_date must not be after end_date")

        observations: list[SPHERExObservation] = []
        for collection in self.collections:
            rows = self._query_collection(
                collection=collection,
                ra=ra_value,
                dec=dec_value,
                radius=radius_value,
                max_results=max_results,
                start_mjd=start_mjd,
                end_mjd=end_mjd,
            )
            observations.extend(self._parse_row(row, collection) for row in rows)

        # Valid times are chronological; records lacking a valid time remain at
        # the end in the archive's stable order.
        observations.sort(
            key=lambda item: (
                item["observation_start_mjd"] is None,
                item["observation_start_mjd"] or 0.0,
            )
        )
        return observations[:max_results]

    @staticmethod
    def _validate_position(ra: float, dec: float, radius: float) -> tuple[float, float, float]:
        values: list[float] = []
        for name, value in (("ra", ra), ("dec", dec), ("radius", radius)):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite number")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be a finite number")
            values.append(numeric)
        ra_value, dec_value, radius_value = values
        if not 0.0 <= ra_value < 360.0:
            raise ValueError("ra must be in the range [0, 360) degrees")
        if not -90.0 <= dec_value <= 90.0:
            raise ValueError("dec must be in the range [-90, 90] degrees")
        if not 0.0 < radius_value <= 180.0:
            raise ValueError("radius must be in the range (0, 180] degrees")
        return ra_value, dec_value, radius_value

    @staticmethod
    def _date_to_mjd(value: DateInput | None, name: str) -> float | None:
        if value is None:
            return None
        candidate: str | datetime
        if isinstance(value, datetime):
            candidate = value
            if candidate.tzinfo is None:
                candidate = candidate.replace(tzinfo=timezone.utc)
        elif isinstance(value, date):
            candidate = datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
        elif isinstance(value, str) and value.strip():
            candidate = value.strip()
        else:
            raise ValueError(f"{name} must be an ISO date/datetime, date, or datetime")
        try:
            return float(Time(candidate, scale="utc").mjd)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} is not a valid date or datetime") from exc

    def _query_collection(
        self,
        *,
        collection: str,
        ra: float,
        dec: float,
        radius: float,
        max_results: int,
        start_mjd: float | None,
        end_mjd: float | None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str | int] = {
            "COLLECTION": collection,
            "POS": f"CIRCLE {ra:.12g} {dec:.12g} {radius:.12g}",
            "MAXREC": max_results,
            "RESPONSEFORMAT": "VOTABLE",
        }
        if start_mjd is not None or end_mjd is not None:
            lower = "-Inf" if start_mjd is None else f"{start_mjd:.12g}"
            upper = "+Inf" if end_mjd is None else f"{end_mjd:.12g}"
            params["TIME"] = f"{lower} {upper}"

        try:
            response = self.session.get(self.endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SPHERExArchiveError(
                f"IRSA SIA2 query failed for collection {collection}: {exc}"
            ) from exc
        if not response.content:
            raise SPHERExArchiveError(
                f"IRSA SIA2 returned an empty response for collection {collection}"
            )

        try:
            # IRSA adds its own harmless formatting hint to FIELD elements.
            # Astropy does not recognize that extension to the VOTable standard.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*Unknown attribute 'irsa_format'.*",
                    category=W48,
                )
                document = parse_votable(BytesIO(response.content), verify="exception")
            for resource in document.resources:
                for info in resource.infos:
                    if str(info.name).upper() == "QUERY_STATUS" and str(info.value).upper() == "ERROR":
                        detail = (info.content or info.value or "unknown SIA2 error").strip()
                        raise SPHERExArchiveError(
                            f"IRSA SIA2 rejected collection {collection}: {detail}"
                        )
            tables = list(document.iter_tables())
            if not tables:
                raise SPHERExArchiveError(
                    f"IRSA SIA2 response for collection {collection} contained no table"
                )
            table = tables[0].to_table(use_names_over_ids=True)
        except SPHERExArchiveError:
            raise
        except Exception as exc:
            raise SPHERExArchiveError(
                f"IRSA SIA2 returned invalid VOTable data for collection {collection}: {exc}"
            ) from exc

        return [
            {column: _plain(row[column]) for column in table.colnames}
            for row in table
        ]

    @staticmethod
    def _parse_row(row: Mapping[str, Any], requested_collection: str) -> SPHERExObservation:
        clean = {str(key): _plain(value) for key, value in row.items()}
        t_min = _number(_first(clean, "t_min", "mjd_obs"))
        t_max = _number(_first(clean, "t_max"))
        timestamp: str | None = None
        if t_min is not None:
            try:
                timestamp = Time(t_min, format="mjd", scale="utc").utc.isot
            except (TypeError, ValueError):
                t_min = None

        access_url = _first(clean, "access_url")
        access_format = _first(clean, "access_format")
        fits_url = SPHERExClient._fits_url(access_url, access_format)
        size_kb = _number(_first(clean, "access_estsize"))
        detector = _first(clean, "detector")
        if detector is None and fits_url:
            match = re.search(r"/level2/[^/]+/[^/]+/([1-6])/[^/]+\.fits(?:$|\?)", fits_url, re.I)
            detector = match.group(1) if match else None

        em_min = _number(_first(clean, "em_min"))
        em_max = _number(_first(clean, "em_max"))
        return {
            "observation_id": _first(clean, "obs_id", "obs_publisher_did"),
            "observation_timestamp": timestamp,
            "observation_start_mjd": t_min,
            "observation_end_mjd": t_max,
            "ra": _number(_first(clean, "s_ra")),
            "dec": _number(_first(clean, "s_dec")),
            "spatial_coverage": _first(clean, "s_region"),
            "field_of_view_deg": _number(_first(clean, "s_fov")),
            "detector": str(detector) if detector is not None else None,
            "band": _first(clean, "energy_bandpassname", "energy_emband", "instrument_name"),
            "wavelength_min_um": em_min * 1e6 if em_min is not None else None,
            "wavelength_max_um": em_max * 1e6 if em_max is not None else None,
            "fits_url": fits_url,
            "exposure_time_s": _number(_first(clean, "t_exptime")),
            "file_size_bytes": round(size_kb * 1024) if size_kb is not None else None,
            "file_size_kb": size_kb,
            "collection": _first(clean, "obs_collection") or requested_collection,
            "access_format": str(access_format) if access_format is not None else None,
            "raw_metadata": clean,
        }

    @staticmethod
    def _fits_url(value: Any, access_format: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        url = value.strip()
        parsed = urlparse(url)
        is_fits_path = parsed.path.lower().endswith((".fits", ".fit", ".fts"))
        is_fits_format = "fits" in str(access_format or "").lower()
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return url if is_fits_path or is_fits_format else None
