"""
Embedding models module.

This module provides embedding models for converting text into dense vector representations.
"""
from ._base import EmbeddingModel
from .openai import OpenAIEmbedding

# Lazy import for SBERTEmbedding to avoid dependency issues
def _get_sbert_embedding():
    """Lazy import for SBERTEmbedding."""
    try:
        from .sbert_embedding import SBERTEmbedding
        return SBERTEmbedding
    except ImportError as e:
        raise ImportError(
            "SBERTEmbedding requires additional dependencies. "
            "Install with: pip install rag-lib[sbert]"
        ) from e

# Export SBERTEmbedding through __getattr__ for lazy loading
def __getattr__(name):
    if name == 'SBERTEmbedding':
        return _get_sbert_embedding()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['EmbeddingModel', 'OpenAIEmbedding', 'SBERTEmbedding']
