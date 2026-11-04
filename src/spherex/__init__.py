"""Small, safe tools for discovering SPHEREx observations at IRSA."""

from .spherex_client import SPHERExArchiveError, SPHERExClient, SPHERExObservation

__version__ = "0.1.0"

__all__ = ["SPHERExArchiveError", "SPHERExClient", "SPHERExObservation"]
