"""
Base class for retrievers.

This module provides the abstract base class for all retrievers in the RAG system.
Retrievers are responsible for finding and ranking relevant documents in response
to user queries by combining document stores and embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import logging

from ..schemas.schema import Document, Query, RetrievalResult
from ..document_stores._base import BaseDocumentStore
from ..models.embedding._base import EmbeddingModel


class RetrieverError(Exception):
    """Base exception for retriever operations."""
    pass


class QueryEncodingError(RetrieverError):
    """Raised when query encoding fails."""
    pass


class RetrievalError(RetrieverError):
    """Raised when document retrieval fails."""
    pass


class BaseRetriever(ABC):
    """
    Abstract base class for retrievers.
    
    Retrievers serve as the bridge between user queries and document stores,
    providing intelligent document retrieval capabilities. They combine:
    - Document stores for storage and basic search
    - Embedding models for semantic understanding
    - Ranking algorithms for result quality
    - Filtering mechanisms for precise results
    
    Key responsibilities:
    - Query processing and encoding
    - Document retrieval from stores
    - Result ranking and filtering
    - Performance optimization
    
    Attributes:
        name (str): Name/identifier of the retriever
        document_store (BaseDocumentStore): Backend document store
        embedding_model (Optional[EmbeddingModel]): Embedding model for query encoding
        config (Dict[str, Any]): Retriever-specific configuration
    """
    
    def __init__(
        self,
        name: str,
        document_store: BaseDocumentStore,
        embedding_model: Optional[EmbeddingModel] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            name (str): Name/identifier of the retriever
            document_store (BaseDocumentStore): Backend document store for retrieval
            embedding_model (Optional[EmbeddingModel]): Embedding model for query encoding
            config (Optional[Dict[str, Any]]): Retriever-specific configuration
        """
        self.name = name
        self.document_store = document_store
        self.embedding_model = embedding_model
        self.config = config or {}
        self._created_at = datetime.now().isoformat()
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}({name})")
        
        # Validate initialization
        self._validate_initialization()
    
    def _validate_initialization(self) -> None:
        """Validate retriever initialization parameters."""
        if not isinstance(self.document_store, BaseDocumentStore):
            raise TypeError("document_store must be an instance of BaseDocumentStore")
        
        if self.embedding_model is not None and not isinstance(self.embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must be an instance of EmbeddingModel")
    
    # Core Retrieval Methods
    
    @abstractmethod
    def retrieve(
        self,
        query: Union[Query, str],
        index_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a given query.
        
        This is the main retrieval method that combines query processing,
        document search, and result ranking.
        
        Args:
            query (Union[Query, str]): User query or Query object
            index_name (str): Name of the index to search in
            top_k (int): Maximum number of results to return
            filters (Optional[Dict[str, Any]]): Metadata filters to apply
            **kwargs: Additional retrieval parameters
            
        Returns:
            List[RetrievalResult]: List of retrieved documents with scores and ranks
            
        Raises:
            QueryEncodingError: If query encoding fails
            RetrievalError: If document retrieval fails
            RetrieverError: If retrieval process fails
        """
        pass
    
    @abstractmethod
    def batch_retrieve(
        self,
        queries: List[Union[Query, str]],
        index_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve relevant documents for multiple queries in batch.
        
        Args:
            queries (List[Union[Query, str]]): List of user queries
            index_name (str): Name of the index to search in
            top_k (int): Maximum number of results per query
            filters (Optional[Dict[str, Any]]): Metadata filters to apply
            **kwargs: Additional retrieval parameters
            
        Returns:
            List[List[RetrievalResult]]: List of retrieval results for each query
            
        Raises:
            QueryEncodingError: If any query encoding fails
            RetrievalError: If batch retrieval fails
        """
        pass
    
    # Query Processing Methods
    
    def process_query(self, query: Union[Query, str]) -> Query:
        """
        Process and validate a query before retrieval.
        
        Args:
            query (Union[Query, str]): Raw query or Query object
            
        Returns:
            Query: Processed Query object
            
        Raises:
            QueryEncodingError: If query processing fails
        """
        try:
            # Convert string to Query object if needed
            if isinstance(query, str):
                if not query.strip():
                    raise QueryEncodingError("Query content cannot be empty")
                query = Query(content=query)
            elif not isinstance(query, Query):
                raise QueryEncodingError("Query must be a string or Query object")
            
            # Validate query content
            if not query.content.strip():
                raise QueryEncodingError("Query content cannot be empty")
            
            # Apply query preprocessing if configured
            processed_query = self._preprocess_query(query)
            
            # Encode query if embedding model is available and embedding is missing
            if self.embedding_model and processed_query.embedding is None:
                processed_query = self.embedding_model.encode_query(processed_query)
            
            return processed_query
            
        except Exception as e:
            if isinstance(e, QueryEncodingError):
                raise
            raise QueryEncodingError(f"Failed to process query: {str(e)}")
    
    def _preprocess_query(self, query: Query) -> Query:
        """
        Apply preprocessing to query content.
        
        This method can be overridden in concrete implementations
        to add query preprocessing like normalization, expansion, etc.
        
        Args:
            query (Query): Input query
            
        Returns:
            Query: Preprocessed query
        """
        # Default implementation: no preprocessing
        return query
    
    # Result Processing Methods
    
    def post_process_results(
        self,
        results: List[RetrievalResult],
        query: Query,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Post-process retrieval results.
        
        This method can be used to apply additional ranking,
        filtering, or result enhancement.
        
        Args:
            results (List[RetrievalResult]): Raw retrieval results
            query (Query): Original query
            **kwargs: Additional processing parameters
            
        Returns:
            List[RetrievalResult]: Post-processed results
        """
        # Apply result filtering if configured
        filtered_results = self._filter_results(results, **kwargs)
        
        # Apply result reranking if configured
        reranked_results = self._rerank_results(filtered_results, query, **kwargs)
        
        # Update result ranks
        for i, result in enumerate(reranked_results):
            result.rank = i + 1
        
        return reranked_results
    
    def _filter_results(
        self,
        results: List[RetrievalResult],
        min_score: Optional[float] = None,
        max_results: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Filter results based on score and count criteria.
        
        Args:
            results (List[RetrievalResult]): Input results
            min_score (Optional[float]): Minimum score threshold
            max_results (Optional[int]): Maximum number of results
            **kwargs: Additional filter parameters
            
        Returns:
            List[RetrievalResult]: Filtered results
        """
        filtered = results
        
        # Apply score filtering
        if min_score is not None:
            filtered = [r for r in filtered if r.score >= min_score]
        
        # Apply count limiting
        if max_results is not None:
            filtered = filtered[:max_results]
        
        return filtered
    
    def _rerank_results(
        self,
        results: List[RetrievalResult],
        query: Query,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Rerank results using additional criteria.
        
        Default implementation maintains original order.
        Override in concrete implementations for custom reranking.
        
        Args:
            results (List[RetrievalResult]): Input results
            query (Query): Original query
            **kwargs: Additional reranking parameters
            
        Returns:
            List[RetrievalResult]: Reranked results
        """
        # Default: no reranking, maintain original order
        return results
    
    # Utility Methods
    
    def get_supported_indices(self) -> List[str]:
        """
        Get list of supported indices from the document store.
        
        Returns:
            List[str]: List of available index names
        """
        try:
            return self.document_store.list_indices()
        except Exception as e:
            self.logger.error(f"Failed to get supported indices: {e}")
            return []
    
    def validate_index(self, index_name: str) -> bool:
        """
        Validate that an index exists and is accessible.
        
        Args:
            index_name (str): Index name to validate
            
        Returns:
            bool: True if index is valid and accessible
        """
        try:
            return self.document_store.index_exists(index_name)
        except Exception as e:
            self.logger.error(f"Failed to validate index '{index_name}': {e}")
            return False
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about an index.
        
        Args:
            index_name (str): Index name
            
        Returns:
            Dict[str, Any]: Index statistics
        """
        try:
            if not self.validate_index(index_name):
                return {"error": f"Index '{index_name}' does not exist"}
            
            info = self.document_store.get_index_info(index_name)
            count = self.document_store.count_documents(index_name)
            
            return {
                "index_name": index_name,
                "document_count": count,
                "index_info": info,
                "retriever_name": self.name,
                "embedding_model": self.embedding_model.get_model_info() if self.embedding_model else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get index stats for '{index_name}': {e}")
            return {"error": str(e)}
    
    def set_embedding_model(self, embedding_model: EmbeddingModel) -> None:
        """
        Set or update the embedding model.
        
        Args:
            embedding_model (EmbeddingModel): New embedding model
        """
        if not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must be an instance of EmbeddingModel")
        
        self.embedding_model = embedding_model
        self.logger.info(f"Updated embedding model to: {embedding_model.model_name}")
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update retriever configuration.
        
        Args:
            config_updates (Dict[str, Any]): Configuration updates to apply
        """
        self.config.update(config_updates)
        self.logger.info(f"Updated configuration: {config_updates}")
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the retriever.
        
        Returns:
            Dict[str, Any]: Retriever information
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "created_at": self._created_at,
            "config": self.config,
            "document_store": {
                "name": self.document_store.name,
                "class": self.document_store.__class__.__name__,
                "indices": self.get_supported_indices()
            },
            "embedding_model": self.embedding_model.get_model_info() if self.embedding_model else None
        }
    
    # Performance and Monitoring
    
    def benchmark_retrieval(
        self,
        queries: List[Union[Query, str]],
        index_name: str,
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark retrieval performance with a set of queries.
        
        Args:
            queries (List[Union[Query, str]]): Test queries
            index_name (str): Index to search in
            top_k (int): Number of results per query
            **kwargs: Additional retrieval parameters
            
        Returns:
            Dict[str, Any]: Benchmark results with timing and statistics
        """
        import time
        
        results = {
            "total_queries": len(queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "total_time": 0.0,
            "average_time_per_query": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "errors": []
        }
        
        query_times = []
        
        start_time = time.time()
        
        for i, query in enumerate(queries):
            query_start = time.time()
            
            try:
                _ = self.retrieve(query, index_name, top_k, **kwargs)
                query_time = time.time() - query_start
                query_times.append(query_time)
                results["successful_queries"] += 1
                
                results["min_time"] = min(results["min_time"], query_time)
                results["max_time"] = max(results["max_time"], query_time)
                
            except Exception as e:
                query_time = time.time() - query_start
                results["failed_queries"] += 1
                results["errors"].append(f"Query {i}: {str(e)}")
                self.logger.error(f"Benchmark query {i} failed: {e}")
        
        results["total_time"] = time.time() - start_time
        
        if query_times:
            results["average_time_per_query"] = sum(query_times) / len(query_times)
        else:
            results["min_time"] = 0.0
        
        return results
    
    def __repr__(self) -> str:
        """String representation of the retriever."""
        store_info = f"{self.document_store.__class__.__name__}({self.document_store.name})"
        embedding_info = f"{self.embedding_model.__class__.__name__}({self.embedding_model.model_name})" if self.embedding_model else "None"
        
        return f"{self.__class__.__name__}(name='{self.name}', store={store_info}, embedding={embedding_info})"
