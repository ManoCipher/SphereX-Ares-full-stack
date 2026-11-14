"""Framework-independent SPHEREx backend."""

from .spherex_client import SPHERExArchiveError, SPHERExClient, SPHERExObservation
from .image_cleaner import SpherexImageCleaner
from .observation_selector import ObservationSelectionService, SelectionMode, SelectionResult

__version__ = "0.1.0"

__all__ = [
    "ObservationSelectionService",
    "SelectionMode",
    "SelectionResult",
    "SPHERExArchiveError",
    "SPHERExClient",
    "SPHERExObservation",
    "SpherexImageCleaner",
]
