"""Framework-neutral service joining archive records to FITS processing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from astropy.io import fits

from .config import ArchiveConfig, ProjectPaths
from .downloader import CutoutDownloader
from .fits_processor import SpherexFitsProcessor


class ObservationPreviewService:
    """Resolve downloaded observations and produce UI-neutral preview records."""

    def __init__(
        self,
        raw_dir: Path | str | None = None,
        preview_dir: Path | str | None = None,
        processor: SpherexFitsProcessor | None = None,
        downloader: CutoutDownloader | None = None,
        download_cutout_size_deg: float = 0.4,
    ) -> None:
        paths = ProjectPaths.default()
        self.raw_dir = Path(raw_dir) if raw_dir is not None else paths.raw
        self.download_dir = self.raw_dir / "viewer"
        self.processor = processor or SpherexFitsProcessor(
            Path(preview_dir) if preview_dir is not None else paths.processed / "previews"
        )
        # A 0.4-degree server-side cutout comfortably contains a 200-pixel
        # SPHEREx preview at the instrument's approximately 6.2 arcsec/pixel
        # plate scale. Only explicitly selected observations are requested.
        self.downloader = downloader or CutoutDownloader(
            ArchiveConfig(cutout_size_deg=download_cutout_size_deg, max_files=6)
        )
        self._file_index: list[Path] | None = None

    def refresh_files(self) -> None:
        """Refresh the downloaded-file index after an external download."""
        self._file_index = None

    def resolve_fits_path(self, observation: Mapping[str, Any]) -> Path | None:
        """Return the local FITS product matching both observation and detector."""
        for container in (observation, observation.get("raw_metadata", {})):
            if not isinstance(container, Mapping):
                continue
            for key in ("file_path", "fits_path", "local_path", "download_path", "filename"):
                value = container.get(key)
                if not value:
                    continue
                candidate = Path(str(value)).expanduser()
                if candidate.is_file():
                    return candidate.resolve()
                if not candidate.is_absolute():
                    matches = [path for path in self._files() if path.name == candidate.name]
                    if matches:
                        return matches[0].resolve()

        observation_id = observation.get("observation_id") or observation.get("obs_id")
        if not observation_id:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(observation_id)).strip("._").casefold()
        exactish = [path for path in self._files() if safe and safe in path.stem.casefold()]
        if not exactish:
            return None

        requested_detector = observation.get("detector")
        if requested_detector not in (None, ""):
            detected = [(path, self._fits_detector(path)) for path in exactish]
            detector_matches = [
                path for path, detector in detected
                if detector == str(requested_detector).casefold()
            ]
            if len(detector_matches) == 1:
                return detector_matches[0].resolve()
            if len(exactish) == 1 and detected[0][1] is None:
                return exactish[0].resolve()
            # A same-ID file for another detector is a different archive product.
            return None
        return exactish[0].resolve() if len(exactish) == 1 else None

    @staticmethod
    def _fits_detector(path: Path) -> str | None:
        try:
            with fits.open(path, mode="readonly", memmap=False) as hdul:
                for key in ("DETECTOR", "DETNAM", "DET_ID", "CHANNEL"):
                    for hdu in hdul:
                        value = hdu.header.get(key)
                        if value not in (None, ""):
                            return str(value).casefold()
        except (OSError, ValueError):
            return None
        return None

    @staticmethod
    def _product_identity(observation: Mapping[str, Any]) -> tuple[str, ...]:
        """Identify one archive product, not merely its shared exposure ID."""
        url = observation.get("fits_url")
        if url:
            return ("url", str(url))
        return (
            "metadata",
            str(observation.get("observation_id") or observation.get("obs_id") or ""),
            str(observation.get("detector") or ""),
            str(observation.get("observation_timestamp") or observation.get("obs_time_iso") or ""),
        )

    @classmethod
    def _validate_unique_products(cls, observations: Sequence[Mapping[str, Any]]) -> None:
        identities = [cls._product_identity(observation) for observation in observations]
        if len(set(identities)) != len(identities):
            raise ValueError("The selection contains the same observation product more than once")

    def download_status(self, observation: Mapping[str, Any]) -> str:
        return "Downloaded" if self.resolve_fits_path(observation) else "Not downloaded"

    def prepare_observation_previews(
        self,
        observations: Sequence[Mapping[str, Any]],
        ra: float,
        dec: float,
        size: int = 200,
        *,
        download_missing: bool = True,
        rendering: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Ensure selected FITS files are local, then create their previews.

        Downloads are bounded to the supplied selection (at most six). Existing
        local files and cached previews are reused.
        """
        if not observations:
            return []
        if len(observations) > 6:
            raise ValueError("Select no more than 6 observations")
        self._validate_unique_products(observations)
        resolved = [dict(observation) for observation in observations]
        if download_missing:
            resolved = self.ensure_local_files(resolved, ra, dec)
        else:
            for record in resolved:
                path = self.resolve_fits_path(record)
                if path is not None:
                    record["file_path"] = str(path)
        return self.processor.prepare_observation_previews(resolved, ra, dec, size, rendering)

    def create_rgb_composite(
        self,
        observations: Sequence[Mapping[str, Any]],
        ra: float,
        dec: float,
        size: int = 200,
        *,
        rendering: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Download three selected records as needed and create an aligned RGB image."""
        if len(observations) != 3:
            raise ValueError("Select exactly 3 observations for an RGB composite")
        self._validate_unique_products(observations)
        resolved = self.ensure_local_files(observations, ra, dec)
        return self.processor.create_rgb_composite(resolved, ra, dec, size, rendering)

    def ensure_local_files(
        self,
        observations: Sequence[Mapping[str, Any]],
        ra: float,
        dec: float,
    ) -> list[dict[str, Any]]:
        """Download only selected records that do not already exist locally."""
        records = [dict(observation) for observation in observations]
        missing: list[dict[str, Any]] = []
        for record in records:
            path = self.resolve_fits_path(record)
            if path is not None:
                record["file_path"] = str(path)
            else:
                missing.append(record)

        downloadable = [record for record in missing if record.get("fits_url")]
        errors: dict[str, str] = {}
        if downloadable:
            rows = pd.DataFrame(
                [
                    {
                        "obs_id": record.get("observation_id") or record.get("obs_id"),
                        "data_release": record.get("collection") or "spherex",
                        "access_url": record.get("fits_url"),
                        "target_ra_deg": float(ra),
                        "target_dec_deg": float(dec),
                    }
                    for record in downloadable
                ]
            )
            manifest = self.downloader.download(
                rows,
                self.download_dir,
                max_files=len(downloadable),
            )
            unmatched = list(downloadable)
            for row in manifest.to_dict(orient="records"):
                observation_id = str(row.get("obs_id"))
                status = str(row.get("status"))
                filename = row.get("filename")
                if status in {"downloaded", "skipped_existing"} and filename:
                    destination = self.download_dir / str(filename)
                    if destination.is_file():
                        source_url = row.get("source_url")
                        matching = next(
                            (record for record in unmatched if source_url and record.get("fits_url") == source_url),
                            None,
                        )
                        if matching is None:
                            matching = next(
                                (
                                    record for record in unmatched
                                    if str(record.get("observation_id") or record.get("obs_id")) == observation_id
                                ),
                                None,
                            )
                        if matching is not None:
                            matching["file_path"] = str(destination.resolve())
                            matching["download_status"] = status
                            unmatched.remove(matching)
                else:
                    errors[observation_id] = str(row.get("error") or f"download status: {status}")
            self.refresh_files()

        for record in missing:
            if record.get("file_path"):
                continue
            observation_id = str(record.get("observation_id") or record.get("obs_id"))
            path = self.resolve_fits_path(record)
            if path is not None:
                record["file_path"] = str(path)
            elif not record.get("fits_url"):
                record["download_error"] = "Observation has no FITS download URL"
            else:
                record["download_error"] = errors.get(observation_id, "FITS download did not produce a local file")
        return records

    def _files(self) -> list[Path]:
        if self._file_index is None:
            extensions = {".fits", ".fit", ".fts"}
            self._file_index = (
                sorted(path for path in self.raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
                if self.raw_dir.is_dir()
                else []
            )
        return self._file_index
