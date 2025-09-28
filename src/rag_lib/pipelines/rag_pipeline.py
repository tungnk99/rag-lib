"""
RAG Pipeline implementation with retriever -> ranker -> generator flow.

This module implements a complete RAG (Retrieval-Augmented Generation) pipeline
that orchestrates the flow from query to final response through:
1. Document retrieval using retrievers
2. Document reranking using rankers  
3. Response generation using generators
"""

from typing import List, Optional, Dict, Any, Union
import time
import logging
from datetime import datetime

from ..schemas.schema import Query, Document, RetrievalResult, RAGResponse, create_query
from ..retrievers._base import BaseRetriever
from ..rankers._base import BaseRanker, NoOpRanker
from ..generators._base import BaseGenerator, GenerationConfig


class RAGPipelineError(Exception):
    """Base exception for RAG pipeline operations."""
    pass


class RetrievalError(RAGPipelineError):
    """Raised when retrieval step fails."""
    pass


class RankingError(RAGPipelineError):
    """Raised when ranking step fails."""
    pass


class GenerationError(RAGPipelineError):
    """Raised when generation step fails."""
    pass


class RAGPipeline:
    """
    Complete RAG pipeline that orchestrates retrieval, ranking, and generation.
    
    The pipeline follows this flow:
    1. Query Processing: Validates and processes the input query
    2. Document Retrieval: Uses retriever to find relevant documents
    3. Document Ranking: Uses ranker to reorder documents by relevance
    4. Response Generation: Uses generator to create final response
    
    Attributes:
        retriever (BaseRetriever): Component for document retrieval
        ranker (BaseRanker): Component for document reranking  
        generator (BaseGenerator): Component for response generation
        config (Dict[str, Any]): Pipeline configuration
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        ranker: Optional[BaseRanker] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever (BaseRetriever): Document retriever
            generator (BaseGenerator): Response generator
            ranker (Optional[BaseRanker]): Document ranker (uses NoOpRanker if None)
            config (Optional[Dict[str, Any]]): Pipeline configuration
        """
        self.retriever = retriever
        self.generator = generator
        self.ranker = ranker or NoOpRanker()
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Validate components
        self._validate_components()
        
        # Pipeline metadata
        self._created_at = datetime.now().isoformat()
        self._query_count = 0
    
    def _validate_components(self) -> None:
        """Validate that all pipeline components are properly configured."""
        if not isinstance(self.retriever, BaseRetriever):
            raise TypeError("retriever must be an instance of BaseRetriever")
        
        if not isinstance(self.generator, BaseGenerator):
            raise TypeError("generator must be an instance of BaseGenerator")
        
        if not isinstance(self.ranker, BaseRanker):
            raise TypeError("ranker must be an instance of BaseRanker")
        
        # Check if generator is available
        if not self.generator.is_available():
            raise RAGPipelineError("Generator is not available or properly configured")
    
    def query(
        self,
        query: Union[str, Query],
        index_name: str,
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
        retrieval_filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query (Union[str, Query]): User query
            index_name (str): Name of the index to search in
            top_k (int): Number of documents to retrieve initially
            rerank_top_k (Optional[int]): Number of documents to keep after reranking
            generation_config (Optional[GenerationConfig]): Generation configuration
            retrieval_filters (Optional[Dict[str, Any]]): Filters for retrieval
            **kwargs: Additional parameters
            
        Returns:
            RAGResponse: Complete response with generated text and metadata
            
        Raises:
            RetrievalError: If document retrieval fails
            RankingError: If document ranking fails  
            GenerationError: If response generation fails
        """
        try:
            start_time = time.time()
            self._query_count += 1
            
            self.logger.info(f"Processing query #{self._query_count}: {query}")
            
            # Step 1: Process query
            processed_query = self._process_query(query)
            
            # Step 2: Retrieve documents
            retrieval_start = time.time()
            retrieval_results = self._retrieve_documents(
                processed_query, index_name, top_k, retrieval_filters, **kwargs
            )
            retrieval_time = time.time() - retrieval_start
            
            # Step 3: Rerank documents
            ranking_start = time.time()
            ranked_results = self._rank_documents(
                retrieval_results, processed_query, rerank_top_k, **kwargs
            )
            ranking_time = time.time() - ranking_start
            
            # Step 4: Generate response
            generation_start = time.time()
            rag_response = self._generate_response(
                processed_query, ranked_results, generation_config, **kwargs
            )
            generation_time = time.time() - generation_start
            
            # Update timing information
            total_time = time.time() - start_time
            rag_response.total_time = total_time
            rag_response.retrieval_time = retrieval_time
            rag_response.generation_time = generation_time
            
            # Add pipeline metadata
            if 'pipeline_metadata' not in rag_response.generation_metadata:
                rag_response.generation_metadata['pipeline_metadata'] = {}
            
            rag_response.generation_metadata['pipeline_metadata'].update({
                'ranking_time': ranking_time,
                'initial_documents': len(retrieval_results),
                'final_documents': len(ranked_results),
                'retriever': self.retriever.__class__.__name__,
                'ranker': self.ranker.__class__.__name__,
                'generator': self.generator.__class__.__name__
            })
            
            self.logger.info(f"Query processed successfully in {total_time:.2f}s")
            return rag_response
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            if isinstance(e, (RetrievalError, RankingError, GenerationError)):
                raise
            else:
                raise RAGPipelineError(f"Unexpected error in pipeline: {str(e)}") from e
    
    def _process_query(self, query: Union[str, Query]) -> Query:
        """
        Process and validate the input query.
        
        Args:
            query (Union[str, Query]): Input query
            
        Returns:
            Query: Processed query object
        """
        if isinstance(query, str):
            processed_query = create_query(query)
        elif isinstance(query, Query):
            processed_query = query
        else:
            raise ValueError("Query must be a string or Query object")
        
        if not processed_query.content.strip():
            raise ValueError("Query content cannot be empty")
        
        return processed_query
    
    def _retrieve_documents(
        self,
        query: Query,
        index_name: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using the retriever.
        
        Args:
            query (Query): Processed query
            index_name (str): Index name
            top_k (int): Number of documents to retrieve
            filters (Optional[Dict[str, Any]]): Retrieval filters
            **kwargs: Additional parameters
            
        Returns:
            List[RetrievalResult]: Retrieved documents with scores
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            self.logger.debug(f"Retrieving {top_k} documents from index '{index_name}'")
            
            results = self.retriever.retrieve(
                query=query,
                index_name=index_name,
                top_k=top_k,
                filters=filters,
                **kwargs
            )
            
            self.logger.debug(f"Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            raise RetrievalError(f"Document retrieval failed: {str(e)}") from e
    
    def _rank_documents(
        self,
        retrieval_results: List[RetrievalResult],
        query: Query,
        rerank_top_k: Optional[int],
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Rerank retrieved documents using the ranker.
        
        Args:
            retrieval_results (List[RetrievalResult]): Initial retrieval results
            query (Query): Original query
            rerank_top_k (Optional[int]): Number of documents to keep after reranking
            **kwargs: Additional parameters
            
        Returns:
            List[RetrievalResult]: Reranked documents
            
        Raises:
            RankingError: If ranking fails
        """
        try:
            if not retrieval_results:
                self.logger.warning("No documents to rank")
                return []
            
            self.logger.debug(f"Ranking {len(retrieval_results)} documents")
            
            # Extract documents from retrieval results
            documents = [result.document for result in retrieval_results]
            
            # Rerank documents
            ranked_results = self.ranker.rerank(
                documents=documents,
                query=query,
                top_k=rerank_top_k,
                **kwargs
            )
            
            self.logger.debug(f"Ranking completed, {len(ranked_results)} documents remain")
            return ranked_results
            
        except Exception as e:
            raise RankingError(f"Document ranking failed: {str(e)}") from e
    
    def _generate_response(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        generation_config: Optional[GenerationConfig],
        **kwargs
    ) -> RAGResponse:
        """
        Generate response using the generator.
        
        Args:
            query (Query): Original query
            retrieval_results (List[RetrievalResult]): Ranked retrieval results
            generation_config (Optional[GenerationConfig]): Generation configuration
            **kwargs: Additional parameters
            
        Returns:
            RAGResponse: Generated response
            
        Raises:
            GenerationError: If generation fails
        """
        try:
            self.logger.debug(f"Generating response with {len(retrieval_results)} documents")
            
            response = self.generator.generate_rag_response(
                query=query,
                retrieval_results=retrieval_results,
                config=generation_config
            )
            
            self.logger.debug("Response generation completed")
            return response
            
        except Exception as e:
            raise GenerationError(f"Response generation failed: {str(e)}") from e
    
    def batch_query(
        self,
        queries: List[Union[str, Query]],
        index_name: str,
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
        retrieval_filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[RAGResponse]:
        """
        Process multiple queries through the pipeline.
        
        Args:
            queries (List[Union[str, Query]]): List of queries to process
            index_name (str): Name of the index to search in
            top_k (int): Number of documents to retrieve initially
            rerank_top_k (Optional[int]): Number of documents to keep after reranking
            generation_config (Optional[GenerationConfig]): Generation configuration
            retrieval_filters (Optional[Dict[str, Any]]): Filters for retrieval
            **kwargs: Additional parameters
            
        Returns:
            List[RAGResponse]: List of responses for each query
        """
        responses = []
        
        for i, query in enumerate(queries):
            self.logger.info(f"Processing batch query {i+1}/{len(queries)}")
            
            try:
                response = self.query(
                    query=query,
                    index_name=index_name,
                    top_k=top_k,
                    rerank_top_k=rerank_top_k,
                    generation_config=generation_config,
                    retrieval_filters=retrieval_filters,
                    **kwargs
                )
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Failed to process batch query {i+1}: {str(e)}")
                # Create error response
                error_response = RAGResponse(
                    query=self._process_query(query),
                    generated_text=f"Error processing query: {str(e)}",
                    retrieved_documents=[],
                    generation_metadata={"error": str(e)}
                )
                responses.append(error_response)
        
        return responses
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dict[str, Any]: Pipeline information
        """
        return {
            "created_at": self._created_at,
            "query_count": self._query_count,
            "retriever": {
                "name": self.retriever.name,
                "class": self.retriever.__class__.__name__,
                "config": getattr(self.retriever, 'config', {})
            },
            "ranker": {
                "class": self.ranker.__class__.__name__,
                "config": getattr(self.ranker, 'config', {})
            },
            "generator": {
                "class": self.generator.__class__.__name__,
                "config": self.generator.config.to_dict() if hasattr(self.generator, 'config') else {}
            },
            "pipeline_config": self.config
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update pipeline configuration.
        
        Args:
            new_config (Dict[str, Any]): New configuration parameters
        """
        self.config.update(new_config)
        self.logger.info("Pipeline configuration updated")
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._query_count = 0
        self.logger.info("Pipeline statistics reset")
    
    def __str__(self) -> str:
        """String representation of the pipeline."""
        return (f"RAGPipeline("
                f"retriever={self.retriever.__class__.__name__}, "
                f"ranker={self.ranker.__class__.__name__}, "
                f"generator={self.generator.__class__.__name__})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the pipeline."""
        return self.__str__()
