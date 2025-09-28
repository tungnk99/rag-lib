"""
Retrievers module for the RAG library.

This module provides document retrieval functionality that bridges user queries
and document stores. Retrievers combine embedding models and document stores
to provide intelligent, context-aware document retrieval.

The module includes:
- BaseRetriever: Abstract base class for all retrievers
- SemanticRetriever: Dense vector similarity-based retrieval
- Custom exceptions for retrieval operations
"""

from ._base import (
    BaseRetriever,
    RetrieverError,
    QueryEncodingError,
    RetrievalError
)
from .semantic import SemanticRetriever

__all__ = [
    "BaseRetriever",
    "SemanticRetriever",
    "RetrieverError",
    "QueryEncodingError", 
    "RetrievalError"
]
