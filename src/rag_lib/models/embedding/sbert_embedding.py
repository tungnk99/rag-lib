"""
SBERT (Sentence-BERT) embedding model implementation.

This module provides SBERT-based embedding models using the sentence-transformers library.
Supports various pre-trained sentence transformer models for generating text embeddings.
"""

import logging
from typing import List, Optional, Dict, Any, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

from ._base import EmbeddingModel

logger = logging.getLogger(__name__)


class SBERTEmbedding(EmbeddingModel):
    """
    SBERT (Sentence-BERT) embedding model implementation.
    
    This class provides access to any sentence transformer model from Hugging Face
    for generating high-quality sentence embeddings. Supports both local inference 
    and GPU acceleration.
    
    Attributes:
        model (SentenceTransformer): The sentence transformer model
        model_name (str): Name of the sentence transformer model
        device (str): Device to run the model on ('cpu', 'cuda', 'mps')
        normalize_embeddings (bool): Whether to normalize embeddings to unit length
        batch_size (int): Batch size for processing multiple texts
        max_seq_length (int): Maximum sequence length for the model
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        max_seq_length: Optional[int] = None,
        similarity_type: str = "cosine",
        trust_remote_code: bool = False,
        cache_folder: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SBERT embedding model.
        
        Args:
            model_name (str): Name of the sentence transformer model (from Hugging Face)
            device (Optional[str]): Device to run model on. If None, auto-detects best device
            normalize_embeddings (bool): Whether to normalize embeddings to unit length
            batch_size (int): Batch size for processing multiple texts
            max_seq_length (Optional[int]): Maximum sequence length. If None, uses model's default
            similarity_type (str): Type of similarity function to use. Options:
                - "cosine": Cosine similarity (default)
                - "dot": Dot product similarity
                - "euclidean": Negative Euclidean distance
                - "manhattan": Negative Manhattan distance
            trust_remote_code (bool): Whether to trust remote code when loading models
            cache_folder (Optional[str]): Custom cache folder for model files
            config (Optional[Dict[str, Any]]): Additional configuration
            
        Raises:
            ImportError: If sentence-transformers package is not installed
            ValueError: If similarity_type is not supported
            Exception: If model loading fails
        """
        if not SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers package not found. "
                "Install it with: pip install sentence-transformers"
            )
        
        # Validate similarity type
        supported_similarities = ["cosine", "dot", "euclidean", "manhattan"]
        if similarity_type not in supported_similarities:
            raise ValueError(
                f"Unsupported similarity_type '{similarity_type}'. "
                f"Supported types: {supported_similarities}"
            )
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.similarity_type = similarity_type
        self.trust_remote_code = trust_remote_code
        self.cache_folder = cache_folder
        
        try:
            # Load the model
            logger.info(f"Loading SBERT model: {model_name} on device: {device}")
            
            model_kwargs = {
                "device": device,
                "trust_remote_code": trust_remote_code
            }
            if cache_folder:
                model_kwargs["cache_folder"] = cache_folder
                
            self.model = SentenceTransformer(model_name, **model_kwargs)
            
            # Get model dimension
            dimension = self.model.get_sentence_embedding_dimension()
            
            # Set max sequence length
            if max_seq_length is not None:
                # Use user-provided max length
                self.max_seq_length = max_seq_length
                # Update model's max_seq_length if possible
                if hasattr(self.model, 'max_seq_length'):
                    self.model.max_seq_length = max_seq_length
            elif hasattr(self.model, 'max_seq_length'):
                # Use model's default max length
                self.max_seq_length = self.model.max_seq_length
            else:
                # Fallback default
                self.max_seq_length = 512
                logger.warning(f"Could not determine max_seq_length for {model_name}, using default: 512")
            
        except Exception as e:
            logger.error(f"Failed to load SBERT model {model_name}: {e}")
            raise Exception(f"Failed to load SBERT model: {e}")
        
        # Initialize base class
        super().__init__(
            model_name=model_name,
            dimension=dimension,
            config=config or {}
        )
        
        logger.info(
            f"Successfully initialized SBERT model: {model_name} "
            f"(dimension: {dimension}, device: {device})"
        )
    
    def validate_input_length(self, text: str) -> None:
        """
        Validate text length against SBERT model limits using actual tokenizer.
        
        Args:
            text (str): Text to validate
            
        Raises:
            ValueError: If text exceeds model limits
        """
        try:
            # Use the model's tokenizer for accurate token counting
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                # Get actual token count using the model's tokenizer
                tokens = self.model.tokenizer(
                    text,
                    add_special_tokens=True,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    truncation=False,
                    padding=False
                )
                actual_token_count = len(tokens['input_ids'])
            else:
                # Fallback to rough estimation if tokenizer not available
                actual_token_count = len(text.split())
                logger.warning("Tokenizer not available, using rough estimation for token counting")
            
            if actual_token_count > self.max_seq_length:
                raise ValueError(
                    f"Token count ({actual_token_count}) exceeds "
                    f"SBERT model limit ({self.max_seq_length}). "
                    f"Consider chunking your text."
                )
                
            logger.debug(f"Text validation passed: {actual_token_count}/{self.max_seq_length} tokens")
            
        except Exception as e:
            if "exceeds" in str(e):
                # Re-raise validation errors
                raise
            else:
                # Log tokenizer errors but don't fail validation
                logger.warning(f"Error during tokenizer validation: {e}. Using fallback validation.")
                # Fallback validation
                estimated_tokens = len(text.split())
                if estimated_tokens > self.max_seq_length:
                    raise ValueError(
                        f"Estimated token count ({estimated_tokens}) exceeds "
                        f"SBERT model limit ({self.max_seq_length}). "
                        f"Consider chunking your text."
                    )
    
    def get_token_count(self, text: str) -> int:
        """
        Get the actual token count for a text using the model's tokenizer.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Number of tokens the text will be tokenized into
        """
        try:
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                tokens = self.model.tokenizer(
                    text,
                    add_special_tokens=True,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    truncation=False,
                    padding=False
                )
                return len(tokens['input_ids'])
            else:
                # Fallback to rough estimation
                return len(text.split())
                
        except Exception as e:
            logger.warning(f"Error getting token count: {e}. Using fallback estimation.")
            return len(text.split())
    
    def encode_text(self, text: str, **kwargs) -> List[float]:
        """
        Encode a single text using SBERT model.
        
        Args:
            text (str): Input text to encode
            **kwargs: Additional parameters for sentence transformer
                - convert_to_numpy (bool): Whether to convert to numpy array
                - show_progress_bar (bool): Whether to show progress bar
                - normalize_embeddings (bool): Override default normalization
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            ValueError: If text is invalid
            Exception: If encoding fails
        """
        # Basic validation
        self._validate_text_input(text)
        self.validate_input_length(text)
        
        try:
            # Override defaults with kwargs
            normalize = kwargs.pop('normalize_embeddings', self.normalize_embeddings)
            convert_to_numpy = kwargs.pop('convert_to_numpy', True)
            show_progress_bar = kwargs.pop('show_progress_bar', False)
            
            # Encode the text
            embedding = self.model.encode(
                text,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress_bar,
                **kwargs
            )
            
            # Convert to list if it's a numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            logger.debug(f"Successfully encoded text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode text with SBERT: {e}")
            raise Exception(f"Failed to encode text: {e}")
    
    def batch_encode_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Encode multiple texts using SBERT model.
        
        Args:
            texts (List[str]): List of input texts to encode
            **kwargs: Additional parameters for sentence transformer
                - batch_size (int): Override default batch size
                - convert_to_numpy (bool): Whether to convert to numpy array
                - show_progress_bar (bool): Whether to show progress bar
                - normalize_embeddings (bool): Override default normalization
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ValueError: If any text is invalid
            Exception: If encoding fails
        """
        # Validate inputs
        self._validate_batch_input(texts)
        for text in texts:
            self.validate_input_length(text)
        
        try:
            # Override defaults with kwargs
            batch_size = kwargs.pop('batch_size', self.batch_size)
            normalize = kwargs.pop('normalize_embeddings', self.normalize_embeddings)
            convert_to_numpy = kwargs.pop('convert_to_numpy', True)
            show_progress_bar = kwargs.pop('show_progress_bar', len(texts) > 100)
            
            # Encode the texts in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress_bar,
                **kwargs
            )
            
            # Convert to list of lists if it's a numpy array
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            logger.debug(f"Successfully encoded {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to batch encode texts with SBERT: {e}")
            raise Exception(f"Failed to encode texts: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dict[str, Any]: Model information including SBERT-specific details
        """
        base_info = super().get_model_info()
        
        sbert_info = {
            "device": self.device,
            "max_seq_length": self.max_seq_length,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "similarity_type": self.similarity_type,
            "provider": "SentenceTransformers",
            "trust_remote_code": self.trust_remote_code,
            "cache_folder": self.cache_folder
        }
        
        # Add model-specific info if available
        if hasattr(self.model, '_modules'):
            try:
                sbert_info["pooling_mode"] = str(self.model._modules.get('1', 'Unknown'))
            except:
                pass
        
        return {**base_info, **sbert_info}
    
   
    def text_similarity(
        self,
        text1: str,
        text2: str, 
        similarity_type: Optional[str] = None
    ) -> float:
        """
        Compute similarity between two texts directly.
        
        Args:
            text1 (str): First text to compare
            text2 (str): Second text to compare
            similarity_type (Optional[str]): Override the default similarity type.
                If None, uses the similarity_type set during initialization.
                Options: "cosine", "dot", "euclidean", "manhattan"
        
        Returns:
            float: Similarity score between the two texts
        
        Example:
            score = model.text_similarity("Hello world", "Hi there")
        """
        try:
            # Validate inputs
            if not isinstance(text1, str) or not isinstance(text2, str):
                raise TypeError("Both inputs must be strings")
            
            if not text1.strip() or not text2.strip():
                raise ValueError("Input texts cannot be empty")
            
            # Use provided similarity type or default to instance setting
            sim_type = similarity_type if similarity_type is not None else self.similarity_type
            
            # Validate similarity type
            supported_similarities = ["cosine", "dot", "euclidean", "manhattan"]
            if sim_type not in supported_similarities:
                raise ValueError(
                    f"Unsupported similarity_type '{sim_type}'. "
                    f"Supported types: {supported_similarities}"
                )
            
            # Encode texts
            embedding1 = self.encode_text(text1)
            embedding2 = self.encode_text(text2)
            
            # Convert to tensors
            emb1 = torch.tensor([embedding1], dtype=torch.float32)
            emb2 = torch.tensor([embedding2], dtype=torch.float32)
            
            # Compute similarity based on type
            if sim_type == "cosine":
                from sentence_transformers.util import cos_sim
                similarity = cos_sim(emb1, emb2)[0, 0]
                
            elif sim_type == "dot":
                # Dot product similarity
                similarity = torch.dot(emb1[0], emb2[0])
                
            elif sim_type == "euclidean":
                # Negative Euclidean distance (higher values = more similar)
                distance = torch.dist(emb1[0], emb2[0], p=2)
                similarity = -distance  # Negative for similarity semantics
                
            elif sim_type == "manhattan":
                # Negative Manhattan distance (higher values = more similar)
                distance = torch.dist(emb1[0], emb2[0], p=1)
                similarity = -distance  # Negative for similarity semantics
            
            return float(similarity)
                
        except Exception as e:
            logger.error(f"Failed to compute text similarity: {e}")
            raise Exception(f"Failed to compute text similarity: {e}")
    
    def __repr__(self) -> str:
        """String representation of the SBERT embedding model."""
        return (
            f"SBERTEmbedding(model='{self.model_name}', "
            f"dimension={self.dimension}, device='{self.device}', "
            f"similarity='{self.similarity_type}')"
        )
