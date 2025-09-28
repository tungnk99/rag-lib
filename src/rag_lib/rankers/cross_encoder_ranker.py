"""
Cross-encoder ranker implementation using sentence-transformers.

This module provides a ranker that uses cross-encoder models from sentence-transformers
to rerank documents based on their relevance to a query. Cross-encoders process
query-document pairs together, typically providing better accuracy than bi-encoders
but at the cost of computational efficiency.
"""

from typing import List, Optional, Dict, Any, Union
import logging
from ..schemas.schema import Document, Query, RetrievalResult
from ._base import BaseRanker

# Set up logging
logger = logging.getLogger(__name__)


class CrossEncoderRanker(BaseRanker):
    """
    A ranker that uses cross-encoder models from sentence-transformers.
    
    Cross-encoders process query-document pairs jointly, which typically
    provides better ranking accuracy compared to bi-encoder approaches,
    but requires more computational resources as each query-document pair
    must be processed separately.
    """
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cross-encoder ranker.
        
        Args:
            model_name (str): Name of the cross-encoder model to use.
                Popular options include:
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, good performance)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2" (better accuracy, slower)
                - "cross-encoder/ms-marco-electra-base" (high accuracy)
            config (Optional[Dict[str, Any]]): Additional configuration parameters.
                Can include:
                - "batch_size": Batch size for processing (default: 32)
                - "max_length": Maximum sequence length (default: 512)
                - "device": Device to run on ("cpu", "cuda", "auto") (default: "auto")
                - "normalize_scores": Whether to normalize scores to [0,1] (default: True)
                - "cache_folder": Path to cache models (default: None)
        """
        super().__init__(config)
        self.model_name = model_name
        self.model = None
        
        # Default configuration
        default_config = {
            "batch_size": 32,
            "max_length": 512,
            "device": "auto",
            "normalize_scores": True,
            "cache_folder": None
        }
        
        # Update with user config
        default_config.update(self.config)
        self.config = default_config
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            
            # Initialize model with configuration
            model_kwargs = {}
            if self.config.get("max_length"):
                model_kwargs["max_length"] = self.config["max_length"]
            if self.config.get("device"):
                model_kwargs["device"] = self.config["device"]
            if self.config.get("cache_folder"):
                model_kwargs["cache_folder"] = self.config["cache_folder"]
            
            self.model = CrossEncoder(
                model_name=self.model_name,
                **model_kwargs
            )
            
            self._is_initialized = True
            logger.info(f"Successfully loaded model: {self.model_name}")
            
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderRanker. "
                "Install it with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Failed to initialize cross-encoder model: {str(e)}") from e
    
    def rerank(self, documents: List[Document], query: Query, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            documents (List[Document]): List of documents to rerank
            query (Query): Query to rank documents against
            top_k (Optional[int]): Number of top documents to return
        
        Returns:
            List[RetrievalResult]: Reranked documents with cross-encoder scores
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If model is not properly initialized
        """
        # Validate inputs
        self.validate_inputs(documents, query)
        
        if not self._is_initialized or self.model is None:
            raise RuntimeError("Cross-encoder model is not properly initialized")
        
        if not documents:
            return []
        
        logger.debug(f"Reranking {len(documents)} documents with query: '{query.content[:100]}...'")
        
        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = []
        for doc in documents:
            # Create query-document pair as expected by cross-encoder
            pair = [query.content, doc.content]
            query_doc_pairs.append(pair)
        
        try:
            # Get scores from cross-encoder
            scores = self.model.predict(
                query_doc_pairs,
                batch_size=self.config.get("batch_size", 32),
                show_progress_bar=False
            )
            
            # Convert numpy array to list if needed
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            # Normalize scores if requested
            if self.config.get("normalize_scores", True):
                scores = self._normalize_scores(scores)
            
            logger.debug(f"Generated {len(scores)} cross-encoder scores")
            
            # Prepare and return results
            return self.prepare_results(documents, scores, query, top_k)
            
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {str(e)}")
            raise RuntimeError(f"Cross-encoder prediction failed: {str(e)}") from e
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using sigmoid activation.
        
        Args:
            scores (List[float]): Raw cross-encoder scores
        
        Returns:
            List[float]: Normalized scores in [0, 1] range
        """
        import math
        
        # Apply sigmoid to normalize scores
        normalized_scores = []
        for score in scores:
            # Sigmoid function: 1 / (1 + exp(-x))
            try:
                normalized_score = 1 / (1 + math.exp(-score))
                normalized_scores.append(normalized_score)
            except OverflowError:
                # Handle extreme values
                if score > 0:
                    normalized_scores.append(1.0)
                else:
                    normalized_scores.append(0.0)
        
        return normalized_scores
    
    def batch_rerank(self, documents_list: List[List[Document]], 
                    queries: List[Query], top_k: Optional[int] = None) -> List[List[RetrievalResult]]:
        """
        Efficiently rerank multiple document lists against multiple queries.
        
        Args:
            documents_list (List[List[Document]]): List of document lists to rerank
            queries (List[Query]): List of queries
            top_k (Optional[int]): Number of top documents to return for each query
        
        Returns:
            List[List[RetrievalResult]]: List of reranked results for each query
        """
        if len(documents_list) != len(queries):
            raise ValueError("documents_list and queries must have the same length")
        
        # For cross-encoders, we process each query-document set separately
        # as they need individual attention mechanisms
        results = []
        for docs, query in zip(documents_list, queries):
            result = self.rerank(docs, query, top_k)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information including name, config, and capabilities
        """
        info = {
            "model_name": self.model_name,
            "is_initialized": self._is_initialized,
            "config": self.config.copy()
        }
        
        if self.model is not None:
            try:
                # Try to get additional model info if available
                if hasattr(self.model, "max_seq_length"):
                    info["max_seq_length"] = self.model.max_seq_length
                if hasattr(self.model, "_target_device"):
                    info["device"] = str(self.model._target_device)
            except Exception:
                # Ignore if we can't get additional info
                pass
        
        return info
    
    def update_model(self, new_model_name: str) -> None:
        """
        Update to a different cross-encoder model.
        
        Args:
            new_model_name (str): Name of the new model to load
        """
        logger.info(f"Updating model from {self.model_name} to {new_model_name}")
        self.model_name = new_model_name
        self._initialize_model()
    
    def __repr__(self) -> str:
        """String representation of the ranker."""
        return f"CrossEncoderRanker(model_name='{self.model_name}', config={self.config})"


class PopularCrossEncoderRankers:
    """
    Factory class for creating popular cross-encoder rankers with predefined configurations.
    """
    
    @staticmethod
    def ms_marco_minilm_l6() -> CrossEncoderRanker:
        """
        Create MS MARCO MiniLM L-6 ranker (fast, good performance).
        
        Returns:
            CrossEncoderRanker: Configured ranker
        """
        return CrossEncoderRanker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            config={"batch_size": 32, "normalize_scores": True}
        )
    
    @staticmethod
    def ms_marco_minilm_l12() -> CrossEncoderRanker:
        """
        Create MS MARCO MiniLM L-12 ranker (better accuracy, slower).
        
        Returns:
            CrossEncoderRanker: Configured ranker
        """
        return CrossEncoderRanker(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            config={"batch_size": 16, "normalize_scores": True}
        )
    
    @staticmethod
    def ms_marco_electra_base() -> CrossEncoderRanker:
        """
        Create MS MARCO Electra Base ranker (high accuracy, slower).
        
        Returns:
            CrossEncoderRanker: Configured ranker
        """
        return CrossEncoderRanker(
            model_name="cross-encoder/ms-marco-electra-base",
            config={"batch_size": 8, "normalize_scores": True}
        )
    
    @staticmethod
    def distilbert_base() -> CrossEncoderRanker:
        """
        Create DistilBERT base ranker (balanced speed/accuracy).
        
        Returns:
            CrossEncoderRanker: Configured ranker
        """
        return CrossEncoderRanker(
            model_name="cross-encoder/ms-marco-distilbert-base-v4",
            config={"batch_size": 24, "normalize_scores": True}
        )
