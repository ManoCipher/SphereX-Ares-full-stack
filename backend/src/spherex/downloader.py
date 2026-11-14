"""Backend FITS cutout downloader."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import requests
from astropy.io import fits
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ArchiveConfig

LOG = logging.getLogger(__name__)
MAX_FILES_HARD_LIMIT = 100
MANIFEST_COLUMNS = [
    "obs_id", "data_release", "filename", "source_url", "cutout_url", "bytes",
    "sha256", "status", "error",
]


def build_cutout_url(access_url: str, ra_deg: float, dec_deg: float, size_deg: float) -> str:
    """Create an IRSA IBE cutout URL, refusing non-cutout-capable inputs."""
    parsed = urlparse(str(access_url))
    if parsed.scheme != "https" or parsed.hostname != "irsa.ipac.caltech.edu":
        raise ValueError("Refusing download: access_url is not IRSA's HTTPS on-prem service")
    if not parsed.path.startswith("/ibe/data/spherex/") or not parsed.path.lower().endswith(".fits"):
        raise ValueError("Refusing full image: URL is not a cutout-capable SPHEREx IBE FITS URL")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({"center": f"{ra_deg:.8f},{dec_deg:.8f}", "size": f"{size_deg:.6f}"})
    return urlunparse(parsed._replace(query=urlencode(query)))


def validate_fits(path: Path) -> None:
    with fits.open(path, mode="readonly", memmap=False, checksum=True) as hdul:
        if len(hdul) == 0:
            raise OSError("FITS file has no HDUs")
        hdul.verify("exception")


def _safe_filename(obs_id: str, release: str, url: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", obs_id).strip("._") or "observation"
    digest = hashlib.sha256(url.encode()).hexdigest()[:10]
    return f"{release}_{stem[:100]}_{digest}_cutout.fits"


class CutoutDownloader:
    def __init__(self, config: ArchiveConfig | None = None, session: requests.Session | None = None) -> None:
        self.config = config or ArchiveConfig()
        self.session = session or requests.Session()
        retry = Retry(
            total=self.config.retries,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def download(
        self,
        observations: pd.DataFrame,
        output_dir: Path,
        *,
        max_files: int | None = None,
        max_mb: float | None = None,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        output_dir.mkdir(parents=True, exist_ok=True)
        max_files = min(max_files if max_files is not None else self.config.max_files, MAX_FILES_HARD_LIMIT)
        budget = int((max_mb if max_mb is not None else self.config.max_mb) * 1024 * 1024)
        used = 0
        manifest: list[dict[str, object]] = []
        seen_urls: set[str] = set()

        for _, row in observations.iterrows():
            if len(manifest) >= max_files:
                break
            base = row.get("access_url")
            record = {
                "obs_id": row.get("obs_id"), "data_release": row.get("data_release"),
                "filename": None, "source_url": base, "cutout_url": None,
                "bytes": 0, "sha256": None, "status": "rejected", "error": None,
            }
            try:
                cutout_url = build_cutout_url(
                    str(base), float(row["target_ra_deg"]), float(row["target_dec_deg"]), self.config.cutout_size_deg
                )
                record["cutout_url"] = cutout_url
                if cutout_url in seen_urls:
                    record.update(status="duplicate", error="duplicate cutout URL in discovery results")
                    manifest.append(record)
                    continue
                seen_urls.add(cutout_url)
                filename = _safe_filename(str(row.get("obs_id")), str(row.get("data_release")), cutout_url)
                destination = output_dir / filename
                record["filename"] = filename
                if destination.exists():
                    validate_fits(destination)
                    record.update(status="skipped_existing", bytes=destination.stat().st_size)
                    manifest.append(record)
                    continue
                if dry_run:
                    record["status"] = "dry_run"
                    manifest.append(record)
                    continue
                if used >= budget:
                    record.update(status="budget_exceeded", error="download budget exhausted")
                    manifest.append(record)
                    break

                partial = destination.with_suffix(destination.suffix + ".part")
                partial.unlink(missing_ok=True)
                digest = hashlib.sha256()
                written = 0
                try:
                    with self.session.get(cutout_url, stream=True, timeout=self.config.timeout_seconds) as response:
                        response.raise_for_status()
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            if used + written + len(chunk) > budget:
                                raise RuntimeError("download would exceed total byte budget")
                            with partial.open("ab") as handle:
                                handle.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                    validate_fits(partial)
                    partial.replace(destination)
                except Exception:
                    partial.unlink(missing_ok=True)
                    raise
                used += written
                record.update(status="downloaded", bytes=written, sha256=digest.hexdigest())
            except Exception as exc:
                record["error"] = str(exc)
                LOG.warning("Could not retrieve %s: %s", row.get("obs_id"), exc)
            manifest.append(record)
        return pd.DataFrame(manifest, columns=MANIFEST_COLUMNS)
