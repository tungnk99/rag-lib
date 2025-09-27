"""
Base class for embedding models.

This module provides the abstract base class for all embedding models in the RAG system.
Embedding models are responsible for converting text into dense vector representations.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any

from ...schemas.schema import Document, Query


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    
    This class defines the interface for all embedding models, whether they are
    API-based (like OpenAI, Cohere) or local models (like SentenceTransformers).
    
    Attributes:
        model_name (str): Name/identifier of the embedding model
        dimension (int): Dimension of the output embeddings
        config (Dict[str, Any]): Model-specific configuration parameters
    """
    
    def __init__(
        self,
        model_name: str,
        dimension: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): Name/identifier of the embedding model
            dimension (int): Dimension of the output embeddings
            config (Optional[Dict[str, Any]]): Model-specific configuration parameters
        """
        self.model_name = model_name
        self.dimension = dimension
        self.config = config or {}
    
    @abstractmethod
    def encode_text(self, text: str, **kwargs) -> List[float]:
        """
        Encode a single text into embedding vector.
        
        Args:
            text (str): Input text to encode
            **kwargs: Additional model-specific parameters
            
        Returns:
            List[float]: Embedding vector of the input text
            
        Raises:
            ValueError: If text is empty or too long
            Exception: If encoding fails
        """
        pass
    
    @abstractmethod
    def batch_encode_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Encode multiple texts into embedding vectors in batch.
        
        Args:
            texts (List[str]): List of input texts to encode
            **kwargs: Additional model-specific parameters
            
        Returns:
            List[List[float]]: List of embedding vectors corresponding to input texts
            
        Raises:
            ValueError: If any text is empty or too long
            Exception: If encoding fails
        """
        pass
    
    def validate_input_length(self, text: str) -> None:
        """
        Validate input text length for model-specific constraints.
        
        Args:
            text (str): Text to validate
            
        Raises:
            ValueError: If text exceeds model limits
            
        Note:
            This is a hook for concrete implementations to add model-specific
            validation like token limits. Default implementation does nothing.
            Override this method in concrete classes as needed.
        """
        # Default implementation does nothing
        # Concrete classes should override this for model-specific validation
        pass
    
    def encode_query(self, query: Query, **kwargs) -> Query:
        """
        Encode a Query object and update its embedding field.
        
        Args:
            query (Query): Query object to encode
            **kwargs: Additional model-specific parameters
            
        Returns:
            Query: Updated query object with embedding
        """
        if not isinstance(query, Query):
            raise TypeError("Input must be a Query object")
        
        # Basic validation + model-specific validation
        self._validate_text_input(query.content)
        self.validate_input_length(query.content)
        
        embedding = self.encode_text(query.content, **kwargs)
        query.embedding = embedding
        return query
    
    def encode_document(self, document: Document, **kwargs) -> Document:
        """
        Encode a Document object and update its embedding field.
        
        Args:
            document (Document): Document object to encode
            **kwargs: Additional model-specific parameters
            
        Returns:
            Document: Updated document object with embedding
        """
        if not isinstance(document, Document):
            raise TypeError("Input must be a Document object")
        
        # Basic validation + model-specific validation
        self._validate_text_input(document.content)
        self.validate_input_length(document.content)
        
        embedding = self.encode_text(document.content, **kwargs)
        document.embedding = embedding
        return document
    
    def batch_encode_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        Encode multiple Document objects in batch.
        
        Args:
            documents (List[Document]): List of Document objects to encode
            **kwargs: Additional model-specific parameters
            
        Returns:
            List[Document]: List of updated document objects with embeddings
        """
        if not all(isinstance(doc, Document) for doc in documents):
            raise TypeError("All inputs must be Document objects")
        
        texts = [doc.content for doc in documents]
        
        # Validate all texts first
        self._validate_batch_input(texts)
        for text in texts:
            self.validate_input_length(text)
        
        embeddings = self.batch_encode_texts(texts, **kwargs)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        return documents
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            int: Embedding dimension
        """
        return self.dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Dict[str, Any]: Model information including name, dimension, config, etc.
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "config": self.config
        }
    
    def _validate_text_input(self, text: str) -> None:
        """
        Validate basic text input before encoding.
        
        Args:
            text (str): Text to validate
            
        Raises:
            ValueError: If text is invalid
            
        Note:
            This only does basic validation (type and empty check).
            Model-specific validation (like token length) should be done 
            in concrete implementations.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
    
    def _validate_batch_input(self, texts: List[str]) -> None:
        """
        Validate batch text input before encoding.
        
        Args:
            texts (List[str]): List of texts to validate
            
        Raises:
            ValueError: If any text is invalid
            
        Note:
            This only does basic validation (type and empty check).
            Model-specific validation (like token length) should be done 
            in concrete implementations.
        """
        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings")
        
        if not texts:
            raise ValueError("Text list cannot be empty")
        
        for i, text in enumerate(texts):
            try:
                self._validate_text_input(text)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid text at index {i}: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of the embedding model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', dimension={self.dimension})"
