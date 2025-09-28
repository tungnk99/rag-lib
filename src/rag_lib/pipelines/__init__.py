"""
Pipelines module for the RAG library.

This module provides pipeline implementations that orchestrate different
components of the RAG system to create complete workflows.

The module includes:
- RAGPipeline: For retrieval-augmented generation with retriever -> ranker -> generator flow
"""

from .rag_pipeline import (
    RAGPipeline,
    RAGPipelineError,
    RetrievalError,
    RankingError,
    GenerationError
)

__all__ = [
    "RAGPipeline", 
    "RAGPipelineError",
    "RetrievalError",
    "RankingError",
    "GenerationError"
]