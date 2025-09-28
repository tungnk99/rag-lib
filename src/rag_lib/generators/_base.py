"""
Base generator class for text generation in RAG systems.

This module provides the abstract base class for all text generators in the RAG system.
Generators are responsible for taking queries and retrieved documents and producing
coherent, contextual responses.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
from ..schemas.schema import Query, Document, RetrievalResult, RAGResponse


class GenerationConfig:
    """
    Configuration class for text generation parameters.
    
    This class encapsulates various settings that control how text is generated,
    such as temperature, max tokens, stop sequences, etc.
    """
    
    def __init__(self,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 top_k: Optional[int] = None,
                 stop_sequences: Optional[List[str]] = None,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 stream: bool = False,
                 **kwargs):
        """
        Initialize generation configuration.
        
        Args:
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (0.0 to 1.0)
            top_p (float): Nucleus sampling parameter
            top_k (Optional[int]): Top-k sampling parameter
            stop_sequences (Optional[List[str]]): Sequences where generation should stop
            frequency_penalty (float): Frequency penalty for repetition
            presence_penalty (float): Presence penalty for repetition
            stream (bool): Whether to stream the response
            **kwargs: Additional configuration parameters
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = stop_sequences or []
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stream = stream
        self.extra_params = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config = {
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'stream': self.stream
        }
        
        if self.top_k is not None:
            config['top_k'] = self.top_k
        
        if self.stop_sequences:
            config['stop_sequences'] = self.stop_sequences
        
        config.update(self.extra_params)
        return config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationConfig':
        """Create configuration from dictionary."""
        return cls(**data)


class GenerationResult:
    """
    Result of a text generation operation.
    
    Contains the generated text along with metadata about the generation process.
    """
    
    def __init__(self,
                 generated_text: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 finish_reason: Optional[str] = None,
                 usage_info: Optional[Dict[str, Any]] = None):
        """
        Initialize generation result.
        
        Args:
            generated_text (str): The generated text
            metadata (Optional[Dict[str, Any]]): Generation metadata
            finish_reason (Optional[str]): Reason why generation stopped
            usage_info (Optional[Dict[str, Any]]): Token usage information
        """
        self.generated_text = generated_text
        self.metadata = metadata or {}
        self.finish_reason = finish_reason
        self.usage_info = usage_info or {}
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'generated_text': self.generated_text,
            'metadata': self.metadata,
            'finish_reason': self.finish_reason,
            'usage_info': self.usage_info,
            'created_at': self.created_at
        }


class BaseGenerator(ABC):
    """
    Abstract base class for all text generators in the RAG system.
    
    This class defines the interface that all concrete generators must implement.
    Generators take queries and retrieved documents as input and produce
    contextual responses.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None, **kwargs):
        """
        Initialize the generator.
        
        Args:
            config (Optional[GenerationConfig]): Generation configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or GenerationConfig()
        self.extra_config = kwargs
        self._is_initialized = False
    
    @abstractmethod
    def generate(self, 
                 query: Query, 
                 documents: List[Document],
                 config: Optional[GenerationConfig] = None) -> GenerationResult:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query (Query): The user's query
            documents (List[Document]): Retrieved documents to use as context
            config (Optional[GenerationConfig]): Override generation config
            
        Returns:
            GenerationResult: Generated response with metadata
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the generator is available and properly configured.
        
        Returns:
            bool: True if generator is ready to use, False otherwise
        """
        pass
    
    def generate_rag_response(self,
                            query: Query,
                            retrieval_results: List[RetrievalResult],
                            config: Optional[GenerationConfig] = None) -> RAGResponse:
        """
        Generate a complete RAG response from query and retrieval results.
        
        Args:
            query (Query): The user's query
            retrieval_results (List[RetrievalResult]): Retrieved documents with scores
            config (Optional[GenerationConfig]): Override generation config
            
        Returns:
            RAGResponse: Complete RAG response with generated text and metadata
        """
        start_time = time.time()
        
        # Extract documents from retrieval results
        documents = [result.document for result in retrieval_results]
        
        # Generate response
        generation_start = time.time()
        generation_result = self.generate(query, documents, config)
        generation_time = time.time() - generation_start
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Create RAG response
        rag_response = RAGResponse(
            query=query,
            generated_text=generation_result.generated_text,
            retrieved_documents=retrieval_results,
            generation_metadata=generation_result.metadata,
            total_time=total_time,
            generation_time=generation_time
        )
        
        return rag_response
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents (List[Document]): Documents to format
            
        Returns:
            str: Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Get document title from metadata or use ID
            title = doc.metadata.get('title', f'Document {doc.id}')
            
            # Format document
            doc_text = f"[Document {i}: {title}]\n{doc.content}\n"
            context_parts.append(doc_text)
        
        return "\n".join(context_parts)
    
    def build_prompt(self, 
                    query: Query, 
                    documents: List[Document],
                    prompt_template: Optional[str] = None) -> str:
        """
        Build a prompt from query and documents using a template.
        
        Args:
            query (Query): The user's query
            documents (List[Document]): Retrieved documents
            prompt_template (Optional[str]): Custom prompt template
            
        Returns:
            str: Formatted prompt
        """
        context = self.format_context(documents)
        
        if prompt_template is None:
            prompt_template = self.get_default_prompt_template()
        
        # Format the prompt
        prompt = prompt_template.format(
            query=query.content,
            context=context,
            query_metadata=query.metadata,
            document_count=len(documents)
        )
        
        return prompt
    
    def get_default_prompt_template(self) -> str:
        """
        Get the default prompt template for this generator.
        
        Returns:
            str: Default prompt template
        """
        return """Based on the following context documents, please answer the question.

Context:
{context}

Question: {query}

Answer: """
    
    def preprocess_query(self, query: Query) -> Query:
        """
        Preprocess the query before generation.
        
        Args:
            query (Query): Original query
            
        Returns:
            Query: Preprocessed query
        """
        # Default implementation returns query as-is
        # Subclasses can override for custom preprocessing
        return query
    
    def postprocess_response(self, 
                           response: str, 
                           query: Query, 
                           documents: List[Document]) -> str:
        """
        Postprocess the generated response.
        
        Args:
            response (str): Raw generated response
            query (Query): Original query
            documents (List[Document]): Retrieved documents
            
        Returns:
            str: Postprocessed response
        """
        # Default implementation returns response as-is
        # Subclasses can override for custom postprocessing
        return response.strip()
    
    def validate_inputs(self, query: Query, documents: List[Document]) -> None:
        """
        Validate inputs before generation.
        
        Args:
            query (Query): Query to validate
            documents (List[Document]): Documents to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(query, Query):
            raise ValueError("Query must be a Query object")
        
        if not query.content.strip():
            raise ValueError("Query content cannot be empty")
        
        if not isinstance(documents, list):
            raise ValueError("Documents must be a list")
        
        for doc in documents:
            if not isinstance(doc, Document):
                raise ValueError("All documents must be Document objects")
    
    def get_generation_metadata(self, 
                              query: Query, 
                              documents: List[Document],
                              config: GenerationConfig) -> Dict[str, Any]:
        """
        Get metadata about the generation process.
        
        Args:
            query (Query): The query
            documents (List[Document]): Retrieved documents
            config (GenerationConfig): Generation configuration
            
        Returns:
            Dict[str, Any]: Generation metadata
        """
        return {
            'generator_class': self.__class__.__name__,
            'generation_config': config.to_dict(),
            'document_count': len(documents),
            'query_length': len(query.content),
            'context_length': len(self.format_context(documents)),
            'timestamp': datetime.now().isoformat()
        }
    
    def __str__(self) -> str:
        """String representation of the generator."""
        return f"{self.__class__.__name__}(config={self.config.to_dict()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the generator."""
        return self.__str__()


# PromptTemplate and templates are now imported from .prompts module
