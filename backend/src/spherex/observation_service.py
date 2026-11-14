"""Framework-neutral service joining archive records to FITS processing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import ProjectPaths
from .fits_processor import SpherexFitsProcessor


class ObservationPreviewService:
    """Resolve downloaded observations and produce UI-neutral preview records."""

    def __init__(
        self,
        raw_dir: Path | str | None = None,
        preview_dir: Path | str | None = None,
        processor: SpherexFitsProcessor | None = None,
    ) -> None:
        paths = ProjectPaths.default()
        self.raw_dir = Path(raw_dir) if raw_dir is not None else paths.raw
        self.processor = processor or SpherexFitsProcessor(
            Path(preview_dir) if preview_dir is not None else paths.processed / "previews"
        )
        self._file_index: list[Path] | None = None

    def refresh_files(self) -> None:
        """Refresh the downloaded-file index after an external download."""
        self._file_index = None

    def resolve_fits_path(self, observation: Mapping[str, Any]) -> Path | None:
        """Return an explicit path or match the observation ID to local files."""
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
        return exactish[0].resolve() if exactish else None

    def download_status(self, observation: Mapping[str, Any]) -> str:
        return "Downloaded" if self.resolve_fits_path(observation) else "Not downloaded"

    def prepare_observation_previews(
        self,
        observations: Sequence[Mapping[str, Any]],
        ra: float,
        dec: float,
        size: int = 200,
    ) -> list[dict[str, Any]]:
        """Resolve local paths and process at most six explicitly selected rows."""
        if not observations:
            return []
        if len(observations) > 6:
            raise ValueError("Select no more than 6 observations")
        resolved: list[dict[str, Any]] = []
        for observation in observations:
            record = dict(observation)
            path = self.resolve_fits_path(observation)
            if path is not None:
                record["file_path"] = str(path)
            resolved.append(record)
        return self.processor.prepare_observation_previews(resolved, ra, dec, size)

    def _files(self) -> list[Path]:
        if self._file_index is None:
            extensions = {".fits", ".fit", ".fts"}
            self._file_index = (
                sorted(path for path in self.raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
                if self.raw_dir.is_dir()
                else []
            )
        return self._file_index
