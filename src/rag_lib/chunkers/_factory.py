"""
Factory for creating chunkers based on chunker type.

This module provides a factory pattern for creating different types of chunkers
based on string identifiers and configuration parameters.
"""

from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import logging

from ._base import BaseChunker
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
    create_semantic_chunker,
    create_openai_semantic_chunker,
    create_sentence_transformer_semantic_chunker
)

logger = logging.getLogger(__name__)


class ChunkerType(Enum):
    """Enum for available chunker types."""
    LENGTH = "length"
    BASE_LENGTH = "base_length"
    RECURSIVE = "recursive"
    RECURSIVE_CODE = "recursive_code"
    RECURSIVE_ACADEMIC = "recursive_academic"
    SEMANTIC = "semantic"
    SEMANTIC_OPENAI = "semantic_openai"
    SEMANTIC_SENTENCE_TRANSFORMER = "semantic_sentence_transformer"


class ChunkerFactory:
    """
    Factory class for creating chunkers based on type and configuration.
    
    This factory supports all available chunker types and provides
    a unified interface for chunker creation.
    """
    
    def __init__(self):
        """Initialize the chunker factory."""
        self._chunker_registry = {
            ChunkerType.LENGTH: self._create_length_chunker,
            ChunkerType.BASE_LENGTH: self._create_base_length_chunker,
            ChunkerType.RECURSIVE: self._create_recursive_chunker,
            ChunkerType.RECURSIVE_CODE: self._create_recursive_code_chunker,
            ChunkerType.RECURSIVE_ACADEMIC: self._create_recursive_academic_chunker,
            ChunkerType.SEMANTIC: self._create_semantic_chunker,
            ChunkerType.SEMANTIC_OPENAI: self._create_semantic_openai_chunker,
            ChunkerType.SEMANTIC_SENTENCE_TRANSFORMER: self._create_semantic_sentence_transformer_chunker
        }
    
    def create_chunker(
        self, 
        chunker_type: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> BaseChunker:
        """
        Create a chunker based on type and configuration.
        
        Args:
            chunker_type (str): Type of chunker to create
            config (Optional[Dict[str, Any]]): Configuration parameters
            
        Returns:
            BaseChunker: Created chunker instance
            
        Raises:
            ValueError: If chunker type is not supported
            TypeError: If configuration is invalid
        """
        if config is None:
            config = {}
        
        # Convert string to enum
        try:
            chunker_enum = ChunkerType(chunker_type.lower())
        except ValueError:
            available_types = [ct.value for ct in ChunkerType]
            raise ValueError(
                f"Unsupported chunker type: {chunker_type}. "
                f"Available types: {available_types}"
            )
        
        # Get factory method
        factory_method = self._chunker_registry[chunker_enum]
        
        try:
            return factory_method(config)
        except Exception as e:
            logger.error(f"Failed to create chunker of type {chunker_type}: {str(e)}")
            raise TypeError(f"Invalid configuration for {chunker_type}: {str(e)}")
    
    def _create_length_chunker(self, config: Dict[str, Any]) -> BaseLengthChunker:
        """Create a length-based chunker."""
        return create_base_length_chunker(**config)
    
    def _create_base_length_chunker(self, config: Dict[str, Any]) -> BaseLengthChunker:
        """Create a base length chunker."""
        return create_base_length_chunker(**config)
    
    def _create_recursive_chunker(self, config: Dict[str, Any]) -> RecursiveChunker:
        """Create a recursive chunker."""
        # Handle custom chunk levels if provided
        if 'chunk_levels' in config and isinstance(config['chunk_levels'], list):
            # Convert dict representations to ChunkLevel objects
            chunk_levels = []
            for level_config in config['chunk_levels']:
                if isinstance(level_config, dict):
                    chunk_levels.append(ChunkLevel(**level_config))
                else:
                    chunk_levels.append(level_config)
            config['chunk_levels'] = chunk_levels
        
        return create_recursive_chunker(**config)
    
    def _create_recursive_code_chunker(self, config: Dict[str, Any]) -> RecursiveChunker:
        """Create a recursive chunker optimized for code."""
        return create_code_recursive_chunker(**config)
    
    def _create_recursive_academic_chunker(self, config: Dict[str, Any]) -> RecursiveChunker:
        """Create a recursive chunker optimized for academic papers."""
        return create_academic_recursive_chunker(**config)
    
    def _create_semantic_chunker(self, config: Dict[str, Any]) -> SemanticChunker:
        """Create a semantic chunker."""
        # Handle embedding model if provided as string
        embedding_model = config.pop('embedding_model', None)
        if isinstance(embedding_model, str):
            embedding_model = self._get_embedding_model(embedding_model)
        
        # Handle semantic config
        semantic_config = config.pop('semantic_config', None)
        if semantic_config:
            if isinstance(semantic_config, dict):
                semantic_config = SemanticChunkConfig(**semantic_config)
            config['config'] = semantic_config
        
        return create_semantic_chunker(
            embedding_model=embedding_model,
            **config
        )
    
    def _create_semantic_openai_chunker(self, config: Dict[str, Any]) -> SemanticChunker:
        """Create a semantic chunker with OpenAI embeddings."""
        api_key = config.pop('api_key', None)
        if not api_key:
            raise ValueError("api_key is required for OpenAI semantic chunker")
        
        model = config.pop('model', 'text-embedding-3-small')
        
        return create_openai_semantic_chunker(
            api_key=api_key,
            model=model,
            **config
        )
    
    def _create_semantic_sentence_transformer_chunker(self, config: Dict[str, Any]) -> SemanticChunker:
        """Create a semantic chunker with SentenceTransformers."""
        model_name = config.pop('model_name', 'all-MiniLM-L6-v2')
        
        return create_sentence_transformer_semantic_chunker(
            model_name=model_name,
            **config
        )
    
    def _get_embedding_model(self, model_name: str) -> Optional[Callable]:
        """Get embedding model by name."""
        if model_name == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                return lambda sentences: model.encode(sentences)
            except ImportError:
                logger.warning("SentenceTransformers not available, using default embedding model")
                return None
        
        elif model_name == "openai":
            logger.warning("OpenAI embedding model requires API key, use semantic_openai type instead")
            return None
        
        else:
            logger.warning(f"Unknown embedding model: {model_name}")
            return None
    
    def get_available_types(self) -> List[str]:
        """Get list of available chunker types."""
        return [ct.value for ct in ChunkerType]
    
    def get_chunker_info(self, chunker_type: str) -> Dict[str, Any]:
        """
        Get information about a chunker type.
        
        Args:
            chunker_type (str): Type of chunker
            
        Returns:
            Dict[str, Any]: Information about the chunker type
        """
        chunker_info = {
            ChunkerType.LENGTH.value: {
                "description": "Token-based chunking with sentence preservation",
                "parameters": ["max_tokens", "overlap_tokens", "preserve_sentences", "tokenizer"],
                "use_cases": ["General text documents", "Simple chunking needs"]
            },
            ChunkerType.BASE_LENGTH.value: {
                "description": "Base length-based chunker with token counting",
                "parameters": ["max_tokens", "overlap_tokens", "preserve_sentences", "tokenizer"],
                "use_cases": ["General text documents", "Simple chunking needs"]
            },
            ChunkerType.RECURSIVE.value: {
                "description": "Hierarchical chunking with multiple levels",
                "parameters": ["chunk_levels", "overlap_tokens", "create_parent_chunks"],
                "use_cases": ["Structured documents", "Complex text hierarchies"]
            },
            ChunkerType.RECURSIVE_CODE.value: {
                "description": "Recursive chunking optimized for code files",
                "parameters": ["overlap_tokens", "create_parent_chunks"],
                "use_cases": ["Source code files", "Programming documentation"]
            },
            ChunkerType.RECURSIVE_ACADEMIC.value: {
                "description": "Recursive chunking optimized for academic papers",
                "parameters": ["overlap_tokens", "create_parent_chunks"],
                "use_cases": ["Research papers", "Academic documents"]
            },
            ChunkerType.SEMANTIC.value: {
                "description": "Similarity-based chunking using sentence embeddings",
                "parameters": ["embedding_model", "similarity_threshold", "max_tokens", "min_tokens"],
                "use_cases": ["Content with clear topics", "Semantic coherence important"]
            },
            ChunkerType.SEMANTIC_OPENAI.value: {
                "description": "Semantic chunking with OpenAI embeddings",
                "parameters": ["api_key", "model", "similarity_threshold", "max_tokens"],
                "use_cases": ["High-quality semantic chunking", "Production systems"]
            },
            ChunkerType.SEMANTIC_SENTENCE_TRANSFORMER.value: {
                "description": "Semantic chunking with SentenceTransformers",
                "parameters": ["model_name", "similarity_threshold", "max_tokens"],
                "use_cases": ["Local semantic chunking", "Offline systems"]
            }
        }
        
        try:
            chunker_enum = ChunkerType(chunker_type.lower())
            return chunker_info[chunker_enum.value]
        except ValueError:
            available_types = [ct.value for ct in ChunkerType]
            raise ValueError(
                f"Unknown chunker type: {chunker_type}. "
                f"Available types: {available_types}"
            )


# Global factory instance
_factory = ChunkerFactory()


def create_chunker(chunker_type: str, config: Optional[Dict[str, Any]] = None) -> BaseChunker:
    """
    Convenience function to create a chunker using the global factory.
    
    Args:
        chunker_type (str): Type of chunker to create
        config (Optional[Dict[str, Any]]): Configuration parameters
        
    Returns:
        BaseChunker: Created chunker instance
        
    Examples:
        >>> # Create a length-based chunker
        >>> chunker = create_chunker("length", {"max_tokens": 512, "overlap_tokens": 50})
        
        >>> # Create a recursive chunker
        >>> chunker = create_chunker("recursive", {"overlap_tokens": 25})
        
        >>> # Create a semantic chunker with SentenceTransformers
        >>> chunker = create_chunker("semantic_sentence_transformer", {
        ...     "model_name": "all-MiniLM-L6-v2",
        ...     "similarity_threshold": 0.6
        ... })
    """
    return _factory.create_chunker(chunker_type, config)


def get_available_chunker_types() -> List[str]:
    """Get list of available chunker types."""
    return _factory.get_available_types()


def get_chunker_type_info(chunker_type: str) -> Dict[str, Any]:
    """Get information about a specific chunker type."""
    return _factory.get_chunker_info(chunker_type)


def create_chunker_from_config(config: Dict[str, Any]) -> BaseChunker:
    """
    Create a chunker from a complete configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration containing 'type' and other parameters
        
    Returns:
        BaseChunker: Created chunker instance
        
    Examples:
        >>> config = {
        ...     "type": "recursive",
        ...     "overlap_tokens": 50,
        ...     "chunk_levels": [
        ...         {
        ...             "name": "section",
        ...             "separators": ["\\n\\n", "# "],
        ...             "max_tokens": 1024,
        ...             "min_tokens": 100
        ...         }
        ...     ]
        ... }
        >>> chunker = create_chunker_from_config(config)
    """
    if 'type' not in config:
        raise ValueError("Configuration must include 'type' field")
    
    chunker_type = config.pop('type')
    return _factory.create_chunker(chunker_type, config)


# Predefined configurations for common use cases
PREDEFINED_CONFIGS = {
    "small_chunks": {
        "type": "length",
        "max_tokens": 256,
        "overlap_tokens": 25,
        "preserve_sentences": True
    },
    "medium_chunks": {
        "type": "length",
        "max_tokens": 512,
        "overlap_tokens": 50,
        "preserve_sentences": True
    },
    "large_chunks": {
        "type": "length",
        "max_tokens": 1024,
        "overlap_tokens": 100,
        "preserve_sentences": True
    },
    "hierarchical_docs": {
        "type": "recursive",
        "overlap_tokens": 50,
        "create_parent_chunks": True
    },
    "code_files": {
        "type": "recursive_code",
        "overlap_tokens": 25,
        "create_parent_chunks": True
    },
    "academic_papers": {
        "type": "recursive_academic",
        "overlap_tokens": 75,
        "create_parent_chunks": True
    },
    "semantic_local": {
        "type": "semantic_sentence_transformer",
        "model_name": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.5,
        "max_tokens": 512,
        "min_tokens": 50
    },
    "semantic_high_quality": {
        "type": "semantic_sentence_transformer",
        "model_name": "all-mpnet-base-v2",
        "similarity_threshold": 0.6,
        "max_tokens": 768,
        "min_tokens": 100
    }
}


def create_predefined_chunker(preset_name: str, **overrides) -> BaseChunker:
    """
    Create a chunker using a predefined configuration.
    
    Args:
        preset_name (str): Name of the predefined configuration
        **overrides: Parameters to override in the preset configuration
        
    Returns:
        BaseChunker: Created chunker instance
        
    Examples:
        >>> # Use medium chunks preset
        >>> chunker = create_predefined_chunker("medium_chunks")
        
        >>> # Use semantic preset with custom threshold
        >>> chunker = create_predefined_chunker("semantic_local", similarity_threshold=0.7)
    """
    if preset_name not in PREDEFINED_CONFIGS:
        available_presets = list(PREDEFINED_CONFIGS.keys())
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {available_presets}"
        )
    
    # Create config with overrides
    config = PREDEFINED_CONFIGS[preset_name].copy()
    config.update(overrides)
    
    return create_chunker_from_config(config)


def get_available_presets() -> List[str]:
    """Get list of available predefined configurations."""
    return list(PREDEFINED_CONFIGS.keys())


def get_preset_info(preset_name: str) -> Dict[str, Any]:
    """Get information about a predefined configuration."""
    if preset_name not in PREDEFINED_CONFIGS:
        available_presets = list(PREDEFINED_CONFIGS.keys())
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {available_presets}"
        )
    
    return PREDEFINED_CONFIGS[preset_name].copy()
