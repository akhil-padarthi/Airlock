"""Airlock - air-gapped question answering over massive PDFs.

Nothing leaves the machine, nothing is kept after the session.
"""

from .config import AirlockConfig
from .pipeline import Airlock, IngestReport, Retrieval
from .provenance import Answer

__all__ = ["Airlock", "AirlockConfig", "Answer", "IngestReport", "Retrieval"]
__version__ = "1.0.0"
