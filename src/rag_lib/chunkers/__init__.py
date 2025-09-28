"""
Chunkers for splitting documents into smaller chunks.

This module provides various chunking strategies for RAG systems:
- BaseLengthChunker: Token-based chunking with sentence preservation
- RecursiveChunker: Hierarchical chunking with multiple levels
- SemanticChunker: Similarity-based chunking using sentence embeddings
"""

from ._base import BaseChunker, ChunkerError, InvalidChunkSizeError
from .base_length_chunker import BaseLengthChunker, create_base_length_chunker
from .recursive_chunker import (
    RecursiveChunker, 
    ChunkLevel, 
    create_recursive_chunker,
    create_code_recursive_chunker,
    create_academic_recursive_chunker
)
from .semantic_chunker import (
    SemanticChunker,
    SemanticChunkConfig,
    SemanticChunkerError,
    create_semantic_chunker,
    create_openai_semantic_chunker,
    create_sentence_transformer_semantic_chunker
)
from ._factory import (
    ChunkerFactory,
    ChunkerType,
    create_chunker,
    get_available_chunker_types,
    get_chunker_type_info,
    create_chunker_from_config,
    create_predefined_chunker,
    get_available_presets,
    get_preset_info,
    PREDEFINED_CONFIGS
)

__all__ = [
    # Base classes
    "BaseChunker",
    "ChunkerError", 
    "InvalidChunkSizeError",
    
    # Length-based chunker
    "BaseLengthChunker",
    "create_base_length_chunker",
    
    # Recursive chunker
    "RecursiveChunker",
    "ChunkLevel",
    "create_recursive_chunker",
    "create_code_recursive_chunker", 
    "create_academic_recursive_chunker",
    
    # Semantic chunker
    "SemanticChunker",
    "SemanticChunkConfig",
    "SemanticChunkerError",
    "create_semantic_chunker",
    "create_openai_semantic_chunker",
    "create_sentence_transformer_semantic_chunker",
    
    # Factory
    "ChunkerFactory",
    "ChunkerType",
    "create_chunker",
    "get_available_chunker_types",
    "get_chunker_type_info",
    "create_chunker_from_config",
    "create_predefined_chunker",
    "get_available_presets",
    "get_preset_info",
    "PREDEFINED_CONFIGS"
]
