"""
Base ranker class for reranking documents based on query relevance.

This module provides the abstract base class for implementing different ranking algorithms
in the RAG system. Rankers take a list of documents and a query, then reorder the documents
based on their relevance to the query.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..schemas.schema import Document, Query, RetrievalResult


class BaseRanker(ABC):
    """
    Abstract base class for document rankers.
    
    A ranker takes a list of documents and a query, then reorders the documents
    based on their relevance to the query. This is typically used after initial
    retrieval to improve the ranking of documents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ranker with optional configuration.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration parameters for the ranker
        """
        self.config = config or {}
        self._is_initialized = False
    
    @abstractmethod
    def rerank(self, documents: List[Document], query: Query, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            documents (List[Document]): List of documents to rerank
            query (Query): Query to rank documents against
            top_k (Optional[int]): Number of top documents to return. If None, return all documents
        
        Returns:
            List[RetrievalResult]: Reranked documents with scores, ordered by relevance (highest first)
        
        Raises:
            ValueError: If documents list is empty or query is invalid
            NotImplementedError: If subclass doesn't implement this method
        """
        pass
    
    def validate_inputs(self, documents: List[Document], query: Query) -> None:
        """
        Validate input documents and query.
        
        Args:
            documents (List[Document]): List of documents to validate
            query (Query): Query to validate
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        if not isinstance(documents, list):
            raise ValueError("Documents must be a list")
        
        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("All items in documents list must be Document objects")
        
        if not isinstance(query, Query):
            raise ValueError("Query must be a Query object")
        
        if not query.content.strip():
            raise ValueError("Query content cannot be empty")
    
    def prepare_results(self, documents: List[Document], scores: List[float], 
                       query: Query, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Prepare RetrievalResult objects from documents and scores.
        
        Args:
            documents (List[Document]): List of documents
            scores (List[float]): List of relevance scores (same order as documents)
            query (Query): Original query
            top_k (Optional[int]): Number of top results to return
        
        Returns:
            List[RetrievalResult]: Sorted retrieval results (highest score first)
        """
        if len(documents) != len(scores):
            raise ValueError("Documents and scores lists must have the same length")
        
        # Create RetrievalResult objects
        results = []
        for doc, score in zip(documents, scores):
            # Ensure score is between 0 and 1
            normalized_score = max(0.0, min(1.0, score))
            result = RetrievalResult(
                document=doc,
                score=normalized_score,
                explanation=f"Reranked by {self.__class__.__name__}"
            )
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results, 1):
            result.rank = i
        
        # Apply top_k limit if specified
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        return results
    
    def batch_rerank(self, documents_list: List[List[Document]], 
                    queries: List[Query], top_k: Optional[int] = None) -> List[List[RetrievalResult]]:
        """
        Rerank multiple document lists against multiple queries.
        
        Args:
            documents_list (List[List[Document]]): List of document lists to rerank
            queries (List[Query]): List of queries (same length as documents_list)
            top_k (Optional[int]): Number of top documents to return for each query
        
        Returns:
            List[List[RetrievalResult]]: List of reranked results for each query
        """
        if len(documents_list) != len(queries):
            raise ValueError("documents_list and queries must have the same length")
        
        results = []
        for docs, query in zip(documents_list, queries):
            result = self.rerank(docs, query, top_k)
            results.append(result)
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dict[str, Any]: Current configuration
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration.
        
        Args:
            new_config (Dict[str, Any]): New configuration parameters
        """
        self.config.update(new_config)
    
    def __repr__(self) -> str:
        """String representation of the ranker."""
        return f"{self.__class__.__name__}(config={self.config})"


class NoOpRanker(BaseRanker):
    """
    A no-operation ranker that returns documents in their original order.
    
    This is useful as a default ranker or for testing purposes.
    All documents receive a score of 0.5.
    """
    
    def rerank(self, documents: List[Document], query: Query, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Return documents in their original order with uniform scores.
        
        Args:
            documents (List[Document]): List of documents to rerank
            query (Query): Query (not used in this implementation)
            top_k (Optional[int]): Number of top documents to return
        
        Returns:
            List[RetrievalResult]: Documents in original order with uniform scores
        """
        self.validate_inputs(documents, query)
        
        # Assign uniform scores
        scores = [0.5] * len(documents)
        
        return self.prepare_results(documents, scores, query, top_k)


class RandomRanker(BaseRanker):
    """
    A ranker that randomly shuffles documents.
    
    This is useful for testing or as a baseline comparison.
    Scores are assigned randomly between 0 and 1.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the random ranker.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration parameters.
                Can include 'seed' for reproducible results.
        """
        super().__init__(config)
        import random
        
        # Set random seed if provided
        seed = self.config.get('seed')
        if seed is not None:
            random.seed(seed)
        
        self.random = random
    
    def rerank(self, documents: List[Document], query: Query, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Randomly rerank documents.
        
        Args:
            documents (List[Document]): List of documents to rerank
            query (Query): Query (not used in this implementation)
            top_k (Optional[int]): Number of top documents to return
        
        Returns:
            List[RetrievalResult]: Randomly ordered documents
        """
        self.validate_inputs(documents, query)
        
        # Assign random scores
        scores = [self.random.random() for _ in documents]
        
        return self.prepare_results(documents, scores, query, top_k)
