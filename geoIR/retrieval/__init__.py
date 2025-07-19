"""Retrieval subpackage: encoder and index building utilities."""
from . import encoder  # noqa: F401
from .index import Index  # noqa: F401

# Re-export for convenience
from .encoder import Encoder  # noqa: F401

# Make Index available at package level
__all__ = ["Encoder", "Index"]
