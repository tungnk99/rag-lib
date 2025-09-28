"""
Semantic retriever implementation.

This module provides a semantic similarity-based retriever that uses
embedding models and vector search for intelligent document retrieval.
"""

from typing import List, Dict, Any, Optional, Union
import time

from ._base import BaseRetriever, RetrievalError, QueryEncodingError
from ..schemas.schema import Document, Query, RetrievalResult
from ..document_stores._base import BaseDocumentStore
from ..models.embedding._base import EmbeddingModel


class SemanticRetriever(BaseRetriever):
    """
    Semantic similarity-based retriever.
    
    This retriever uses embedding models to encode queries and performs
    semantic similarity search in the document store. It provides:
    - Dense vector retrieval using embeddings
    - Configurable similarity thresholds
    - Metadata filtering capabilities
    - Result post-processing and reranking
    
    Configuration options:
    - similarity_threshold: Minimum similarity score for results
    - normalize_scores: Whether to normalize similarity scores
    - enable_reranking: Whether to apply additional reranking
    - query_expansion: Whether to expand queries
    """
    
    def __init__(
        self,
        name: str,
        document_store: BaseDocumentStore,
        embedding_model: Optional[EmbeddingModel] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the semantic retriever.
        
        Args:
            name (str): Name/identifier of the retriever
            document_store (BaseDocumentStore): Backend document store
            embedding_model (Optional[EmbeddingModel]): Embedding model for encoding
            config (Optional[Dict[str, Any]]): Retriever configuration
        """
        # Set default config
        default_config = {
            "similarity_threshold": 0.0,
            "normalize_scores": True,
            "enable_reranking": False,
            "query_expansion": False,
            "max_query_length": 512,
            "enable_caching": False
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, document_store, embedding_model, default_config)
        
        # Validate that embedding model is available for semantic search
        if self.embedding_model is None:
            self.logger.warning(
                "No embedding model provided. Semantic retrieval will require "
                "pre-encoded queries or will fall back to document store query method."
            )
    
    def retrieve(
        self,
        query: Union[Query, str],
        index_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using semantic similarity.
        
        Args:
            query (Union[Query, str]): User query
            index_name (str): Name of the index to search in
            top_k (int): Maximum number of results to return
            filters (Optional[Dict[str, Any]]): Metadata filters
            **kwargs: Additional parameters (similarity_threshold, etc.)
            
        Returns:
            List[RetrievalResult]: Ranked retrieval results
        """
        try:
            start_time = time.time()
            
            # Validate index
            if not self.validate_index(index_name):
                raise RetrievalError(f"Index '{index_name}' does not exist or is not accessible")
            
            # Process query
            processed_query = self.process_query(query)
            
            # Extract parameters
            similarity_threshold = kwargs.get('similarity_threshold', self.config.get('similarity_threshold', 0.0))
            
            # Perform retrieval
            if processed_query.embedding is not None:
                # Use semantic search with embeddings
                results = self._semantic_search(
                    processed_query, index_name, top_k, filters, similarity_threshold, **kwargs
                )
            else:
                # Fall back to document store query method
                self.logger.warning("No query embedding available, falling back to document store query")
                results = self._fallback_search(
                    processed_query, index_name, top_k, filters, similarity_threshold, **kwargs
                )
            
            # Post-process results
            final_results = self.post_process_results(results, processed_query, **kwargs)
            
            # Log performance
            retrieval_time = time.time() - start_time
            self.logger.debug(
                f"Retrieved {len(final_results)} documents for query '{processed_query.content[:50]}...' "
                f"in {retrieval_time:.3f}s"
            )
            
            return final_results
            
        except Exception as e:
            if isinstance(e, (RetrievalError, QueryEncodingError)):
                raise
            raise RetrievalError(f"Semantic retrieval failed: {str(e)}")
    
    def batch_retrieve(
        self,
        queries: List[Union[Query, str]],
        index_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve relevant documents for multiple queries.
        
        Args:
            queries (List[Union[Query, str]]): List of user queries
            index_name (str): Name of the index to search in
            top_k (int): Maximum number of results per query
            filters (Optional[Dict[str, Any]]): Metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List[List[RetrievalResult]]: Results for each query
        """
        try:
            start_time = time.time()
            
            # Validate index
            if not self.validate_index(index_name):
                raise RetrievalError(f"Index '{index_name}' does not exist or is not accessible")
            
            results = []
            successful_queries = 0
            
            for i, query in enumerate(queries):
                try:
                    query_results = self.retrieve(query, index_name, top_k, filters, **kwargs)
                    results.append(query_results)
                    successful_queries += 1
                except Exception as e:
                    self.logger.error(f"Failed to retrieve for query {i}: {e}")
                    results.append([])  # Empty results for failed query
            
            # Log batch performance
            batch_time = time.time() - start_time
            self.logger.info(
                f"Batch retrieval completed: {successful_queries}/{len(queries)} successful "
                f"in {batch_time:.3f}s (avg: {batch_time/len(queries):.3f}s per query)"
            )
            
            return results
            
        except Exception as e:
            if isinstance(e, RetrievalError):
                raise
            raise RetrievalError(f"Batch retrieval failed: {str(e)}")
    
    def _semantic_search(
        self,
        query: Query,
        index_name: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        similarity_threshold: float,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Perform semantic search using query embeddings.
        
        Args:
            query (Query): Processed query with embeddings
            index_name (str): Index to search in
            top_k (int): Maximum results
            filters (Optional[Dict[str, Any]]): Metadata filters
            similarity_threshold (float): Minimum similarity score
            **kwargs: Additional parameters
            
        Returns:
            List[RetrievalResult]: Search results
        """
        try:
            # Use document store's query method for semantic search
            results = self.document_store.query(
                query=query,
                index_name=index_name,
                top_k=top_k,
                filters=filters,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
            
            # Normalize scores if configured
            if self.config.get('normalize_scores', True):
                results = self._normalize_scores(results)
            
            return results
            
        except Exception as e:
            raise RetrievalError(f"Semantic search failed: {str(e)}")
    
    def _fallback_search(
        self,
        query: Query,
        index_name: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        similarity_threshold: float,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Fallback search when no query embedding is available.
        
        This method tries to use the document store's query capabilities
        or falls back to metadata filtering.
        
        Args:
            query (Query): Query without embeddings
            index_name (str): Index to search in
            top_k (int): Maximum results
            filters (Optional[Dict[str, Any]]): Metadata filters
            similarity_threshold (float): Minimum similarity score
            **kwargs: Additional parameters
            
        Returns:
            List[RetrievalResult]: Search results
        """
        try:
            # Try document store query (might work for some stores)
            try:
                results = self.document_store.query(
                    query=query,
                    index_name=index_name,
                    top_k=top_k,
                    filters=filters,
                    similarity_threshold=similarity_threshold,
                    **kwargs
                )
                return results
                
            except Exception:
                # Fall back to metadata filtering if available
                if filters:
                    documents = self.document_store.filter_documents(
                        index_name=index_name,
                        filters=filters,
                        limit=top_k
                    )
                    
                    # Convert to RetrievalResult with default scores
                    results = []
                    for i, doc in enumerate(documents):
                        result = RetrievalResult(
                            document=doc,
                            score=1.0 - (i * 0.1),  # Decreasing scores
                            rank=i + 1,
                            explanation="Metadata filtering fallback"
                        )
                        results.append(result)
                    
                    return results
                else:
                    # No filters available, return empty results
                    self.logger.warning("No embeddings or filters available for retrieval")
                    return []
                    
        except Exception as e:
            raise RetrievalError(f"Fallback search failed: {str(e)}")
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Normalize similarity scores to [0, 1] range.
        
        Args:
            results (List[RetrievalResult]): Input results
            
        Returns:
            List[RetrievalResult]: Results with normalized scores
        """
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        score_range = max_score - min_score
        if score_range == 0:
            return results
        
        # Normalize scores
        for result in results:
            result.score = (result.score - min_score) / score_range
            result.explanation = f"Normalized score: {result.score:.4f}"
        
        return results
    
    def _preprocess_query(self, query: Query) -> Query:
        """
        Apply semantic retriever specific query preprocessing.
        
        Args:
            query (Query): Input query
            
        Returns:
            Query: Preprocessed query
        """
        # Apply query expansion if enabled
        if self.config.get('query_expansion', False):
            query = self._expand_query(query)
        
        # Truncate query if too long
        max_length = self.config.get('max_query_length', 512)
        if len(query.content) > max_length:
            query.content = query.content[:max_length]
            self.logger.warning(f"Query truncated to {max_length} characters")
        
        return query
    
    def _expand_query(self, query: Query) -> Query:
        """
        Expand query with additional terms.
        
        This is a placeholder for query expansion techniques.
        Override in subclasses for specific expansion strategies.
        
        Args:
            query (Query): Original query
            
        Returns:
            Query: Expanded query
        """
        # Simple expansion: add common synonyms or related terms
        # This is a basic implementation - use more sophisticated methods in practice
        
        expansion_rules = {
            "AI": "artificial intelligence machine learning",
            "ML": "machine learning artificial intelligence",
            "NLP": "natural language processing text analysis",
            "python": "programming language development coding"
        }
        
        expanded_content = query.content
        for term, expansion in expansion_rules.items():
            if term.lower() in query.content.lower():
                expanded_content += f" {expansion}"
        
        if expanded_content != query.content:
            query.content = expanded_content
            query.metadata["expanded"] = True
            self.logger.debug(f"Query expanded: {query.content}")
        
        return query
    
    def _rerank_results(
        self,
        results: List[RetrievalResult],
        query: Query,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Apply semantic-specific reranking.
        
        Args:
            results (List[RetrievalResult]): Input results
            query (Query): Original query
            **kwargs: Reranking parameters
            
        Returns:
            List[RetrievalResult]: Reranked results
        """
        if not self.config.get('enable_reranking', False):
            return results
        
        # Apply additional reranking based on metadata or content
        # This is a simple example - implement more sophisticated reranking as needed
        
        reranking_factors = kwargs.get('reranking_factors', {})
        
        for result in results:
            original_score = result.score
            
            # Boost scores based on metadata
            boost = 0.0
            
            # Boost recent documents
            if 'created_at' in result.document.metadata:
                # Simple recency boost (placeholder)
                boost += 0.1
            
            # Boost documents with certain categories
            preferred_categories = reranking_factors.get('preferred_categories', [])
            if (preferred_categories and 
                result.document.metadata.get('category') in preferred_categories):
                boost += 0.2
            
            # Apply boost
            result.score = min(1.0, original_score + boost)
            
            if boost > 0:
                result.explanation = f"Original: {original_score:.4f}, Boosted: {result.score:.4f}"
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the embedding dimension from the embedding model.
        
        Returns:
            Optional[int]: Embedding dimension or None if no model
        """
        if self.embedding_model:
            return self.embedding_model.get_embedding_dimension()
        return None
    
    def __repr__(self) -> str:
        """String representation of the semantic retriever."""
        embedding_dim = self.get_embedding_dimension()
        return (f"SemanticRetriever(name='{self.name}', "
                f"store={self.document_store.__class__.__name__}, "
                f"embedding_dim={embedding_dim}, "
                f"threshold={self.config.get('similarity_threshold', 0.0)})")
