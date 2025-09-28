"""
Rankers module for the RAG library.

This module provides different ranking algorithms to reorder documents
based on their relevance to a query. Rankers are typically used after
initial retrieval to improve the ranking of documents.
"""

from ._base import BaseRanker, NoOpRanker
from .cross_encoder_ranker import CrossEncoderRanker, PopularCrossEncoderRankers

__all__ = [
    "BaseRanker",
    "NoOpRanker", 
    "CrossEncoderRanker",
    "PopularCrossEncoderRankers"
]
