"""
In-memory document store implementation.

This module provides an in-memory implementation of the BaseDocumentStore for testing
and development purposes. All data is stored in memory and will be lost when the 
application terminates.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import numpy as np
from collections import defaultdict
import copy

from ._base import (
    BaseDocumentStore,
    DocumentStoreError,
    IndexNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentError
)
from ..schemas.schema import Document, Query, RetrievalResult
from ..models.embedding._base import EmbeddingModel


class InMemoryDocumentStore(BaseDocumentStore):
    """
    In-memory implementation of DocumentStore.
    
    This implementation stores all documents and indices in memory using Python
    dictionaries and lists. It's suitable for testing, development, and small datasets.
    
    Features:
    - Fast operations due to in-memory storage
    - Full-text search capabilities
    - Metadata filtering
    - Semantic similarity search with embeddings
    - Complete CRUD operations
    
    Limitations:
    - Data is lost when application terminates
    - Limited by available RAM
    - Not suitable for production with large datasets
    """
    
    def __init__(
        self,
        name: str = "inmemory_store",
        config: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize the InMemoryDocumentStore.
        
        Args:
            name (str): Name of the document store
            config (Optional[Dict[str, Any]]): Configuration parameters
            embedding_model (Optional[EmbeddingModel]): Embedding model for automatic embedding generation
        """
        super().__init__(name, config, embedding_model)
        
        # Storage structures
        self._indices: Dict[str, Dict[str, Any]] = {}  # index_name -> index_info
        self._documents: Dict[str, Dict[str, Document]] = {}  # index_name -> {doc_id -> document}
        self._metadata_index: Dict[str, Dict[str, List[str]]] = {}  # index_name -> {metadata_key -> [doc_ids]}
        
    # Index Management Methods
    
    def create_index(self, index_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new index for storing documents."""
        if index_name in self._indices:
            raise DocumentStoreError(f"Index '{index_name}' already exists")
        
        self._indices[index_name] = {
            "name": index_name,
            "config": config or {},
            "created_at": datetime.now().isoformat(),
            "document_count": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        self._documents[index_name] = {}
        self._metadata_index[index_name] = defaultdict(list)
        
        return True
    
    def delete_index(self, index_name: str, force: bool = False) -> bool:
        """Delete an existing index and all its documents."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        doc_count = len(self._documents.get(index_name, {}))
        if doc_count > 0 and not force:
            raise DocumentStoreError(
                f"Index '{index_name}' contains {doc_count} documents. "
                "Use force=True to delete anyway."
            )
        
        del self._indices[index_name]
        del self._documents[index_name]
        del self._metadata_index[index_name]
        
        return True
    
    def list_indices(self) -> List[str]:
        """List all available indices."""
        return list(self._indices.keys())
    
    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        return index_name in self._indices
    
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get information about an index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        info = self._indices[index_name].copy()
        info["document_count"] = len(self._documents[index_name])
        return info
    
    # Document CRUD Operations
    
    def add_document(
        self,
        document: Document,
        index_name: str,
        auto_embed: bool = True,
        overwrite: bool = False
    ) -> bool:
        """Add a document to the specified index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        if document.id in self._documents[index_name] and not overwrite:
            raise DuplicateDocumentError(f"Document '{document.id}' already exists in index '{index_name}'")
        
        # Create a copy to avoid external modifications
        doc_copy = copy.deepcopy(document)
        
        # Auto-generate embedding if requested and not present
        if auto_embed and doc_copy.embedding is None and self.embedding_model:
            doc_copy = self.embedding_model.encode_document(doc_copy)
        
        # Update metadata timestamps
        doc_copy.metadata["updated_at"] = datetime.now().isoformat()
        
        # Store document
        self._documents[index_name][document.id] = doc_copy
        
        # Update metadata index
        self._update_metadata_index(index_name, doc_copy)
        
        # Update index info
        self._indices[index_name]["last_updated"] = datetime.now().isoformat()
        
        return True
    
    def add_documents(
        self,
        documents: List[Document],
        index_name: str,
        auto_embed: bool = True,
        overwrite: bool = False,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Add multiple documents to the specified index in batch."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        added_count = 0
        failed_count = 0
        errors = []
        
        for i, document in enumerate(documents):
            try:
                self.add_document(document, index_name, auto_embed, overwrite)
                added_count += 1
            except Exception as e:
                failed_count += 1
                errors.append(f"Document {i} (ID: {document.id}): {str(e)}")
        
        return {
            "added_count": added_count,
            "failed_count": failed_count,
            "total_count": len(documents),
            "errors": errors
        }
    
    def get_document(self, document_id: str, index_name: str) -> Document:
        """Retrieve a document by ID from the specified index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        if document_id not in self._documents[index_name]:
            raise DocumentNotFoundError(f"Document '{document_id}' not found in index '{index_name}'")
        
        return copy.deepcopy(self._documents[index_name][document_id])
    
    def get_documents(
        self,
        document_ids: List[str],
        index_name: str,
        include_missing: bool = False
    ) -> List[Document]:
        """Retrieve multiple documents by IDs from the specified index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        documents = []
        for doc_id in document_ids:
            if doc_id in self._documents[index_name]:
                documents.append(copy.deepcopy(self._documents[index_name][doc_id]))
            elif include_missing:
                documents.append(None)
        
        return documents
    
    def update_document(
        self,
        document: Document,
        index_name: str,
        auto_embed: bool = True
    ) -> bool:
        """Update an existing document in the specified index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        if document.id not in self._documents[index_name]:
            raise DocumentNotFoundError(f"Document '{document.id}' not found in index '{index_name}'")
        
        # Remove old metadata index entries
        old_doc = self._documents[index_name][document.id]
        self._remove_from_metadata_index(index_name, old_doc)
        
        # Update document (similar to add_document)
        return self.add_document(document, index_name, auto_embed, overwrite=True)
    
    def delete_document(self, document_id: str, index_name: str) -> bool:
        """Delete a document from the specified index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        if document_id not in self._documents[index_name]:
            raise DocumentNotFoundError(f"Document '{document_id}' not found in index '{index_name}'")
        
        # Remove from metadata index
        doc = self._documents[index_name][document_id]
        self._remove_from_metadata_index(index_name, doc)
        
        # Remove document
        del self._documents[index_name][document_id]
        
        # Update index info
        self._indices[index_name]["last_updated"] = datetime.now().isoformat()
        
        return True
    
    def delete_documents(
        self,
        document_ids: List[str],
        index_name: str,
        ignore_missing: bool = True
    ) -> Dict[str, Any]:
        """Delete multiple documents from the specified index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        deleted_count = 0
        failed_count = 0
        errors = []
        
        for doc_id in document_ids:
            try:
                self.delete_document(doc_id, index_name)
                deleted_count += 1
            except DocumentNotFoundError as e:
                if not ignore_missing:
                    failed_count += 1
                    errors.append(str(e))
            except Exception as e:
                failed_count += 1
                errors.append(f"Document {doc_id}: {str(e)}")
        
        return {
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "total_count": len(document_ids),
            "errors": errors
        }
    
    def document_exists(self, document_id: str, index_name: str) -> bool:
        """Check if a document exists in the specified index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        return document_id in self._documents[index_name]
    
    def count_documents(
        self, 
        index_name: str, 
        filters: Optional[Dict[str, Any]] = None,
        return_embeddings: Optional[bool] = None
    ) -> int:
        """Count documents in the specified index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        documents = list(self._documents[index_name].values())
        
        # Apply filters
        if filters:
            documents = self._apply_metadata_filters(documents, filters)
        
        # Apply embedding filter
        if return_embeddings is not None:
            if return_embeddings:
                documents = [doc for doc in documents if doc.embedding is not None]
            else:
                documents = [doc for doc in documents if doc.embedding is None]
        
        return len(documents)
    
    # Query and Retrieval Methods
    
    def query(
        self,
        query: Union[Query, str],
        index_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0,
        **kwargs
    ) -> List[RetrievalResult]:
        """Query documents in the specified index using semantic similarity."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        # Validate and normalize query
        if isinstance(query, str):
            query = Query(content=query)
        
        # Generate query embedding if not present
        if query.embedding is None:
            if self.embedding_model is None:
                raise DocumentStoreError("No embedding model available for query encoding")
            query = self.embedding_model.encode_query(query)
        
        # Get candidate documents
        documents = list(self._documents[index_name].values())
        
        # Apply metadata filters
        if filters:
            documents = self._apply_metadata_filters(documents, filters)
        
        # Filter documents with embeddings
        documents = [doc for doc in documents if doc.embedding is not None]
        
        if not documents:
            return []
        
        # Calculate similarities
        results = []
        for doc in documents:
            similarity = self._calculate_cosine_similarity(query.embedding, doc.embedding)
            
            if similarity >= similarity_threshold:
                result = RetrievalResult(
                    document=copy.deepcopy(doc),
                    score=similarity,
                    explanation=f"Cosine similarity: {similarity:.4f}"
                )
                results.append(result)
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def filter_documents(
        self,
        index_name: str,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        offset: int = 0,
        return_embeddings: Optional[bool] = None
    ) -> List[Document]:
        """Filter documents by metadata criteria."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        documents = list(self._documents[index_name].values())
        
        # Apply metadata filters
        documents = self._apply_metadata_filters(documents, filters)
        
        # Apply embedding filter
        if return_embeddings is not None:
            if return_embeddings:
                documents = [doc for doc in documents if doc.embedding is not None]
            else:
                documents = [doc for doc in documents if doc.embedding is None]
        
        # Apply pagination
        documents = documents[offset:]
        if limit is not None:
            documents = documents[:limit]
        
        return [copy.deepcopy(doc) for doc in documents]
    
    # Embedding Management Methods
    
    def update_embeddings(
        self,
        index_name: str,
        documents: Optional[List[Document]] = None,
        batch_size: int = 100,
        force: bool = False
    ) -> Dict[str, Any]:
        """Update embeddings for documents in the specified index."""
        if index_name not in self._indices:
            raise IndexNotFoundError(f"Index '{index_name}' does not exist")
        
        if self.embedding_model is None:
            raise DocumentStoreError("No embedding model available")
        
        # Get documents to update
        if documents is None:
            # Update all documents in index
            target_docs = list(self._documents[index_name].values())
        else:
            # Update specific documents
            target_docs = []
            for doc in documents:
                if doc.id in self._documents[index_name]:
                    target_docs.append(self._documents[index_name][doc.id])
        
        # Filter documents that need embedding updates
        if not force:
            target_docs = [doc for doc in target_docs if doc.embedding is None]
        
        updated_count = 0
        failed_count = 0
        errors = []
        
        # Process in batches
        for i in range(0, len(target_docs), batch_size):
            batch = target_docs[i:i + batch_size]
            
            try:
                # Update embeddings for batch
                updated_batch = self.embedding_model.batch_encode_documents(copy.deepcopy(batch))
                
                # Store updated documents
                for updated_doc in updated_batch:
                    self._documents[index_name][updated_doc.id] = updated_doc
                    updated_count += 1
                    
            except Exception as e:
                failed_count += len(batch)
                errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
        
        return {
            "updated_count": updated_count,
            "failed_count": failed_count,
            "total_candidates": len(target_docs),
            "errors": errors
        }
    
    # Helper Methods
    
    def _update_metadata_index(self, index_name: str, document: Document) -> None:
        """Update metadata index for a document."""
        for key, value in document.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                value_str = str(value)
                if document.id not in self._metadata_index[index_name][f"{key}:{value_str}"]:
                    self._metadata_index[index_name][f"{key}:{value_str}"].append(document.id)
    
    def _remove_from_metadata_index(self, index_name: str, document: Document) -> None:
        """Remove document from metadata index."""
        for key, value in document.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                value_str = str(value)
                metadata_key = f"{key}:{value_str}"
                if metadata_key in self._metadata_index[index_name]:
                    if document.id in self._metadata_index[index_name][metadata_key]:
                        self._metadata_index[index_name][metadata_key].remove(document.id)
    
    def _apply_metadata_filters(
        self, 
        documents: List[Document], 
        filters: Dict[str, Any]
    ) -> List[Document]:
        """Apply metadata filters to a list of documents."""
        if not filters:
            return documents
        
        filtered_docs = []
        for doc in documents:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            
            # Ensure result is in [0, 1] range (for semantic similarity)
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception:
            return 0.0
    
    def __repr__(self) -> str:
        """String representation of the in-memory document store."""
        total_docs = sum(len(docs) for docs in self._documents.values())
        return f"InMemoryDocumentStore(name='{self.name}', indices={len(self._indices)}, total_docs={total_docs})"
