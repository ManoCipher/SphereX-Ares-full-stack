"""Backend configuration and project paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


CURRENT_WIDE_COLLECTIONS = ("spherex_dr1", "spherex_qr3")


@dataclass(frozen=True)
class ArchiveConfig:
    sia_url: str = "https://irsa.ipac.caltech.edu/SIA"
    collections: tuple[str, ...] = CURRENT_WIDE_COLLECTIONS
    search_radius_deg: float = 0.05
    cutout_size_deg: float = 0.02
    timeout_seconds: float = 60.0
    retries: int = 4
    max_files: int = 10
    max_mb: float = 300.0


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path = field(init=False)
    processed: Path = field(init=False)
    metadata: Path = field(init=False)
    plots: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", self.root / "data" / "raw")
        object.__setattr__(self, "processed", self.root / "data" / "processed")
        object.__setattr__(self, "metadata", self.root / "data" / "metadata")
        object.__setattr__(self, "plots", self.root / "outputs" / "plots")

    @classmethod
    def default(cls) -> "ProjectPaths":
        return cls(Path(__file__).resolve().parents[3])

    def ensure(self) -> None:
        for directory in (self.raw, self.processed, self.metadata, self.plots):
            directory.mkdir(parents=True, exist_ok=True)
