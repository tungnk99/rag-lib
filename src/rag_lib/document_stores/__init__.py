"""
Document stores module for the RAG library.

This module provides document storage and retrieval functionality for the RAG system.
It includes abstract base classes and concrete implementations for various document stores.
"""

from ._base import (
    BaseDocumentStore,
    DocumentStoreError,
    IndexNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentError
)
from .inmemory import InMemoryDocumentStore

# Optional imports with graceful degradation
try:
    from .qdrant import QdrantDocumentStore
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

__all__ = [
    "BaseDocumentStore",
    "InMemoryDocumentStore",
    "DocumentStoreError", 
    "IndexNotFoundError",
    "DocumentNotFoundError",
    "DuplicateDocumentError"
]

if _QDRANT_AVAILABLE:
    __all__.append("QdrantDocumentStore")
