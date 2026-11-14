"""Framework-independent SPHEREx backend."""

from .spherex_client import SPHERExArchiveError, SPHERExClient, SPHERExObservation
from .image_cleaner import SpherexImageCleaner

__version__ = "0.1.0"

__all__ = ["SPHERExArchiveError", "SPHERExClient", "SPHERExObservation", "SpherexImageCleaner"]
