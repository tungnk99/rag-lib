"""
Base class for document stores.

This module provides the abstract base class for all document stores in the RAG system.
Document stores are responsible for storing, retrieving, and managing documents with their embeddings.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime
import uuid

from ..schemas.schema import Document, Query, RetrievalResult
from ..models.embedding._base import EmbeddingModel


class DocumentStoreError(Exception):
    """Base exception for document store operations."""
    pass


class IndexNotFoundError(DocumentStoreError):
    """Raised when trying to access a non-existent index."""
    pass


class DocumentNotFoundError(DocumentStoreError):
    """Raised when trying to access a non-existent document."""
    pass


class DuplicateDocumentError(DocumentStoreError):
    """Raised when trying to add a document that already exists."""
    pass


class BaseDocumentStore(ABC):
    """
    Abstract base class for document stores.
    
    This class defines the interface for all document stores, providing:
    - Index management (create, delete, list indices)
    - Document CRUD operations (add, update, delete, get)
    - Query functionality for document retrieval
    - Embedding management and updates
    
    Attributes:
        name (str): Name/identifier of the document store
        config (Dict[str, Any]): Store-specific configuration parameters
        embedding_model (Optional[EmbeddingModel]): Embedding model for automatic embedding generation
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize the document store.
        
        Args:
            name (str): Name/identifier of the document store
            config (Optional[Dict[str, Any]]): Store-specific configuration parameters
            embedding_model (Optional[EmbeddingModel]): Embedding model for automatic embedding generation
        """
        self.name = name
        self.config = config or {}
        self.embedding_model = embedding_model
        self._created_at = datetime.now().isoformat()
    
    # Index Management Methods
    
    @abstractmethod
    def create_index(self, index_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new index for storing documents.
        
        Args:
            index_name (str): Name of the index to create
            config (Optional[Dict[str, Any]]): Index-specific configuration
            
        Returns:
            bool: True if index was created successfully
            
        Raises:
            DocumentStoreError: If index creation fails
        """
        pass
    
    @abstractmethod
    def delete_index(self, index_name: str, force: bool = False) -> bool:
        """
        Delete an existing index and all its documents.
        
        Args:
            index_name (str): Name of the index to delete
            force (bool): If True, delete even if index contains documents
            
        Returns:
            bool: True if index was deleted successfully
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            DocumentStoreError: If deletion fails or index contains documents and force=False
        """
        pass
    
    @abstractmethod
    def list_indices(self) -> List[str]:
        """
        List all available indices.
        
        Returns:
            List[str]: List of index names
        """
        pass
    
    @abstractmethod
    def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.
        
        Args:
            index_name (str): Name of the index to check
            
        Returns:
            bool: True if index exists
        """
        pass
    
    @abstractmethod
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """
        Get information about an index.
        
        Args:
            index_name (str): Name of the index
            
        Returns:
            Dict[str, Any]: Index information (document count, config, created_at, etc.)
            
        Raises:
            IndexNotFoundError: If index doesn't exist
        """
        pass
    
    # Document CRUD Operations
    
    @abstractmethod
    def add_document(
        self,
        document: Document,
        index_name: str,
        auto_embed: bool = True,
        overwrite: bool = False
    ) -> bool:
        """
        Add a document to the specified index.
        
        Args:
            document (Document): Document to add
            index_name (str): Name of the index to add document to
            auto_embed (bool): If True, automatically generate embedding if not present
            overwrite (bool): If True, overwrite existing document with same ID
            
        Returns:
            bool: True if document was added successfully
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            DuplicateDocumentError: If document exists and overwrite=False
            DocumentStoreError: If addition fails
        """
        pass
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        index_name: str,
        auto_embed: bool = True,
        overwrite: bool = False,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Add multiple documents to the specified index in batch.
        
        Args:
            documents (List[Document]): List of documents to add
            index_name (str): Name of the index to add documents to
            auto_embed (bool): If True, automatically generate embeddings if not present
            overwrite (bool): If True, overwrite existing documents with same IDs
            batch_size (int): Number of documents to process in each batch
            
        Returns:
            Dict[str, Any]: Summary of batch operation (added_count, failed_count, errors, etc.)
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            DocumentStoreError: If batch addition fails
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str, index_name: str) -> Document:
        """
        Retrieve a document by ID from the specified index.
        
        Args:
            document_id (str): ID of the document to retrieve
            index_name (str): Name of the index to search in
            
        Returns:
            Document: Retrieved document
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            DocumentNotFoundError: If document doesn't exist
        """
        pass
    
    @abstractmethod
    def get_documents(
        self,
        document_ids: List[str],
        index_name: str,
        include_missing: bool = False
    ) -> List[Document]:
        """
        Retrieve multiple documents by IDs from the specified index.
        
        Args:
            document_ids (List[str]): List of document IDs to retrieve
            index_name (str): Name of the index to search in
            include_missing (bool): If True, include None for missing documents
            
        Returns:
            List[Document]: List of retrieved documents
            
        Raises:
            IndexNotFoundError: If index doesn't exist
        """
        pass
    
    @abstractmethod
    def update_document(
        self,
        document: Document,
        index_name: str,
        auto_embed: bool = True
    ) -> bool:
        """
        Update an existing document in the specified index.
        
        Args:
            document (Document): Updated document (must have existing ID)
            index_name (str): Name of the index containing the document
            auto_embed (bool): If True, automatically update embedding
            
        Returns:
            bool: True if document was updated successfully
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            DocumentNotFoundError: If document doesn't exist
            DocumentStoreError: If update fails
        """
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str, index_name: str) -> bool:
        """
        Delete a document from the specified index.
        
        Args:
            document_id (str): ID of the document to delete
            index_name (str): Name of the index containing the document
            
        Returns:
            bool: True if document was deleted successfully
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            DocumentNotFoundError: If document doesn't exist
        """
        pass
    
    @abstractmethod
    def delete_documents(
        self,
        document_ids: List[str],
        index_name: str,
        ignore_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Delete multiple documents from the specified index.
        
        Args:
            document_ids (List[str]): List of document IDs to delete
            index_name (str): Name of the index containing the documents
            ignore_missing (bool): If True, don't raise error for missing documents
            
        Returns:
            Dict[str, Any]: Summary of deletion operation (deleted_count, failed_count, errors, etc.)
            
        Raises:
            IndexNotFoundError: If index doesn't exist
        """
        pass
    
    @abstractmethod
    def document_exists(self, document_id: str, index_name: str) -> bool:
        """
        Check if a document exists in the specified index.
        
        Args:
            document_id (str): ID of the document to check
            index_name (str): Name of the index to search in
            
        Returns:
            bool: True if document exists
            
        Raises:
            IndexNotFoundError: If index doesn't exist
        """
        pass
    
    @abstractmethod
    def count_documents(
        self, 
        index_name: str, 
        filters: Optional[Dict[str, Any]] = None,
        return_embeddings: Optional[bool] = None
    ) -> int:
        """
        Count documents in the specified index.
        
        Args:
            index_name (str): Name of the index to count documents in
            filters (Optional[Dict[str, Any]]): Optional filters to apply
            return_embeddings (Optional[bool]): If True, only count docs with embeddings;
                                               If False, only count docs without embeddings;
                                               If None, count all docs regardless of embeddings
            
        Returns:
            int: Number of documents matching the criteria
            
        Raises:
            IndexNotFoundError: If index doesn't exist
        """
        pass
    
    # Query and Retrieval Methods
    
    @abstractmethod
    def query(
        self,
        query: Union[Query, str],
        index_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Query documents in the specified index using semantic similarity.
        
        Args:
            query (Union[Query, str]): Query object or query string
            index_name (str): Name of the index to search in
            top_k (int): Maximum number of results to return
            filters (Optional[Dict[str, Any]]): Metadata filters to apply
            similarity_threshold (float): Minimum similarity score for results
            **kwargs: Additional query parameters
            
        Returns:
            List[RetrievalResult]: List of retrieval results sorted by relevance
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            DocumentStoreError: If query fails
        """
        pass
    
    
    @abstractmethod
    def filter_documents(
        self,
        index_name: str,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        offset: int = 0,
        return_embeddings: Optional[bool] = None
    ) -> List[Document]:
        """
        Filter documents by metadata criteria.
        
        Args:
            index_name (str): Name of the index to search in
            filters (Dict[str, Any]): Metadata filters to apply
            limit (Optional[int]): Maximum number of results to return
            offset (int): Number of results to skip
            return_embeddings (Optional[bool]): If True, only return docs with embeddings;
                                               If False, only return docs without embeddings;
                                               If None, return all docs regardless of embeddings
            
        Returns:
            List[Document]: List of documents matching the filters
            
        Raises:
            IndexNotFoundError: If index doesn't exist
        """
        pass
    
    # Embedding Management Methods
    
    @abstractmethod
    def update_embeddings(
        self,
        index_name: str,
        documents: Optional[List[Document]] = None,
        batch_size: int = 100,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Update embeddings for documents in the specified index.
        
        Args:
            index_name (str): Name of the index containing documents to update
            document_ids (Optional[List[str]]): Specific document IDs to update (None for all)
            embedding_model (Optional[EmbeddingModel]): Embedding model to use (None for default)
            batch_size (int): Number of documents to process in each batch
            force (bool): If True, update even if embeddings already exist
            
        Returns:
            Dict[str, Any]: Summary of update operation (updated_count, failed_count, errors, etc.)
            
        Raises:
            IndexNotFoundError: If index doesn't exist
            DocumentStoreError: If update fails
        """
        pass
    
    
    
    def get_store_info(self) -> Dict[str, Any]:
        """
        Get information about the document store.
        
        Returns:
            Dict[str, Any]: Store information including name, config, indices, etc.
        """
        return {
            "name": self.name,
            "config": self.config,
            "created_at": self._created_at,
            "indices": self.list_indices()
        }
    
    def __repr__(self) -> str:
        """String representation of the document store."""
        return f"{self.__class__.__name__}(name='{self.name}', indices={len(self.list_indices())})"
