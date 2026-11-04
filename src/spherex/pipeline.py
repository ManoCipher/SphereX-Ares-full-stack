from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from astropy.time import Time

from .archive_client import ArchiveQueryError, IrsaSiaClient
from .config import ArchiveConfig, ProjectPaths
from .downloader import CutoutDownloader
from .fits_processor import inspect_directory
from .scientific_visualization import analyze_directory
from .targets import get_target

LOG = logging.getLogger("spherex")


def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    except (ImportError, ValueError):
        LOG.info("Parquet engine unavailable; CSV output was saved")


def _config(args: argparse.Namespace) -> ArchiveConfig:
    collections = (args.collection,) if getattr(args, "collection", None) else ("spherex_qr3", "spherex_qr2")
    return ArchiveConfig(
        collections=collections,
        search_radius_deg=getattr(args, "search_radius_deg", 0.05),
        cutout_size_deg=getattr(args, "cutout_size_deg", 0.02),
        max_files=getattr(args, "max_files", 10),
        max_mb=getattr(args, "max_mb", 300.0),
    )


def _discover(args: argparse.Namespace, paths: ProjectPaths) -> pd.DataFrame:
    target = get_target(args.target)
    config = _config(args)
    epoch = Time(args.epoch) if args.epoch else Time.now()
    frame = IrsaSiaClient(config).discover(target, epoch=epoch)
    _save(frame, paths.metadata / f"{target.key}_observations.csv")
    print(f"Discovered {len(frame)} record(s); saved metadata for {target.key}.")
    if not frame.empty:
        print(frame[["obs_id", "obs_time_iso", "data_release", "band_min_um", "band_max_um"]].head(20).to_string(index=False))
    return frame


def command_discover(args: argparse.Namespace, paths: ProjectPaths) -> int:
    _discover(args, paths)
    return 0


def command_download(args: argparse.Namespace, paths: ProjectPaths) -> int:
    observations = _discover(args, paths)
    if observations.empty:
        print("No matching observations; nothing to download.")
        return 0
    config = _config(args)
    manifest = CutoutDownloader(config).download(
        observations,
        paths.raw / args.target,
        max_files=args.max_files,
        max_mb=args.max_mb,
        dry_run=args.dry_run,
    )
    _save(manifest, paths.metadata / f"{args.target}_manifest.csv")
    print(manifest[["obs_id", "filename", "status", "bytes", "error"]].to_string(index=False))
    return 0 if not (manifest.get("status", pd.Series(dtype=str)) == "rejected").all() else 2


def command_inspect(args: argparse.Namespace, paths: ProjectPaths) -> int:
    if args.dry_run:
        files = list((paths.raw / args.target).glob("*.fits"))
        print(f"Would inspect {len(files)} FITS file(s).")
        return 0
    report = inspect_directory(paths.raw / args.target, paths.plots / args.target)
    _save(report, paths.metadata / f"{args.target}_fits_inventory.csv")
    print(f"Inspected {report['filename'].nunique() if not report.empty else 0} FITS file(s), {len(report)} HDU record(s).")
    return 0


def command_science(args: argparse.Namespace, paths: ProjectPaths) -> int:
    """Create scientific dashboards only from FITS files already on disk."""
    raw_dir = paths.raw / args.target
    files = list(raw_dir.glob("*.fits"))
    if args.dry_run:
        print(f"Would analyze {len(files)} existing FITS file(s); no downloads would be made.")
        return 0
    target = get_target(args.target)
    summary_path = paths.root / "outputs" / f"{target.key}_observation_summary.csv"
    frame = analyze_directory(
        raw_dir,
        paths.plots / target.key,
        summary_path,
        target,
        metadata_path=paths.metadata / f"{target.key}_observations.csv",
    )
    inside = int(frame.get("target_inside", pd.Series(dtype=bool)).fillna(False).sum())
    evidence = int((frame.get("detection_status", pd.Series(dtype=str)) == "EVIDENCE").sum())
    print(f"Analyzed {len(frame)} existing FITS file(s): {inside} target footprints, {evidence} candidate evidence result(s).")
    print(f"Summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe SPHEREx discovery and cutout pipeline")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--target", default="wise0855")
        command.add_argument("--dry-run", action="store_true")

    discover = sub.add_parser("discover", help="query IRSA SIA2 and save observation metadata")
    common(discover)
    discover.add_argument("--collection", choices=("spherex_qr3", "spherex_qr2"))
    discover.add_argument("--epoch", help="ISO epoch for the initial moving-target search center")
    discover.add_argument("--search-radius-deg", type=float, default=0.05)
    discover.set_defaults(func=command_discover)

    download = sub.add_parser("download", help="discover and download bounded IRSA cutouts")
    common(download)
    download.add_argument("--collection", choices=("spherex_qr3", "spherex_qr2"))
    download.add_argument("--epoch")
    download.add_argument("--search-radius-deg", type=float, default=0.05)
    download.add_argument("--cutout-size-deg", type=float, default=0.02)
    download.add_argument("--max-files", type=int, default=10)
    download.add_argument("--max-mb", type=float, default=300.0)
    download.set_defaults(func=command_download)

    inspect = sub.add_parser("inspect", help="inventory FITS HDUs and make image diagnostics")
    common(inspect)
    inspect.set_defaults(func=command_inspect)

    science = sub.add_parser("science", help="make WCS-aware scientific dashboards from existing cutouts")
    common(science)
    science.set_defaults(func=command_science)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    paths = ProjectPaths.default()
    paths.ensure()
    try:
        return args.func(args, paths)
    except (ArchiveQueryError, ValueError, OSError) as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
