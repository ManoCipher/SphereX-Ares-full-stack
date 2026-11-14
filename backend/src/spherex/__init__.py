"""Framework-independent SPHEREx backend."""

from .spherex_client import SPHERExArchiveError, SPHERExClient, SPHERExObservation

__version__ = "0.1.0"

__all__ = ["SPHERExArchiveError", "SPHERExClient", "SPHERExObservation"]
