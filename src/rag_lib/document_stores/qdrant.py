"""
Qdrant document store implementation.

This module provides a Qdrant-based implementation of the BaseDocumentStore for
production-ready vector search and document management. Qdrant is a high-performance
vector database optimized for similarity search.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import json

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, CreateCollection, UpdateCollection,
        PointStruct, Filter, FieldCondition, Range, MatchValue,
        ScoredPoint, SearchRequest, CountRequest, ScrollRequest,
        UpdateStatus, CollectionInfo
    )
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    raise ImportError(
        "qdrant-client is required for QdrantDocumentStore. "
        "Install it with: pip install qdrant-client"
    )

from ._base import (
    BaseDocumentStore,
    DocumentStoreError,
    IndexNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentError
)
from ..schemas.schema import Document, Query, RetrievalResult
from ..models.embedding._base import EmbeddingModel


class QdrantDocumentStore(BaseDocumentStore):
    """
    Qdrant-based implementation of DocumentStore.
    
    This implementation uses Qdrant as the backend vector database for storing
    documents and their embeddings. It provides high-performance semantic search
    and supports advanced filtering capabilities.
    
    Features:
    - High-performance vector similarity search
    - Advanced metadata filtering with Qdrant conditions
    - Scalable storage for large document collections
    - HNSW indexing for fast approximate nearest neighbor search
    - Persistent storage with backup and recovery options
    - Support for different distance metrics
    
    Requirements:
    - Qdrant server running (local or cloud)
    - qdrant-client Python package
    """
    
    def __init__(
        self,
        name: str = "qdrant_store",
        config: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        qdrant_timeout: int = 60,
        default_vector_size: int = 768,
        default_distance: Distance = Distance.COSINE
    ):
        """
        Initialize the QdrantDocumentStore.
        
        Args:
            name (str): Name of the document store
            config (Optional[Dict[str, Any]]): Configuration parameters
            embedding_model (Optional[EmbeddingModel]): Embedding model for automatic embedding generation
            qdrant_url (str): URL of the Qdrant server
            qdrant_api_key (Optional[str]): API key for Qdrant cloud
            qdrant_timeout (int): Timeout for Qdrant operations in seconds
            default_vector_size (int): Default vector size for new collections
            default_distance (Distance): Default distance metric for similarity search
        """
        super().__init__(name, config, embedding_model)
        
        # Qdrant configuration
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_timeout = qdrant_timeout
        self.default_vector_size = default_vector_size
        self.default_distance = default_distance
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=qdrant_timeout
            )
            # Test connection
            self.client.get_collections()
        except Exception as e:
            raise DocumentStoreError(f"Failed to connect to Qdrant at {qdrant_url}: {str(e)}")
    
    # Index Management Methods (Collections in Qdrant)
    
    def create_index(self, index_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection in Qdrant for storing documents."""
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            for collection in collections:
                if collection.name == index_name:
                    raise DocumentStoreError(f"Collection '{index_name}' already exists")
            
            # Parse config
            config = config or {}
            vector_size = config.get('vector_size', self.default_vector_size)
            distance = config.get('distance', self.default_distance)
            
            # Validate vector size
            if not isinstance(vector_size, int) or vector_size <= 0:
                raise DocumentStoreError(f"Invalid vector_size: {vector_size}. Must be positive integer.")
            
            # Create collection
            self.client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            # Store collection metadata
            collection_metadata = {
                "created_at": datetime.now().isoformat(),
                "vector_size": vector_size,
                "distance": distance.value,
                "config": config
            }
            
            # Store metadata as a special point
            self.client.upsert(
                collection_name=index_name,
                points=[PointStruct(
                    id="__metadata__",
                    vector=[0.0] * vector_size,  # Dummy vector
                    payload={"__is_metadata__": True, "metadata": collection_metadata}
                )]
            )
            
            return True
            
        except Exception as e:
            raise DocumentStoreError(f"Failed to create collection '{index_name}': {str(e)}")
    
    def delete_index(self, index_name: str, force: bool = False) -> bool:
        """Delete an existing collection and all its documents."""
        try:
            # Check if collection exists
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            # Check if collection has documents (excluding metadata)
            if not force:
                count_result = self.client.count(
                    collection_name=index_name,
                    count_filter=Filter(
                        must_not=[FieldCondition(key="__is_metadata__", match=MatchValue(value=True))]
                    )
                )
                doc_count = count_result.count
                
                if doc_count > 0:
                    raise DocumentStoreError(
                        f"Collection '{index_name}' contains {doc_count} documents. "
                        "Use force=True to delete anyway."
                    )
            
            # Delete collection
            self.client.delete_collection(collection_name=index_name)
            return True
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to delete collection '{index_name}': {str(e)}")
    
    def list_indices(self) -> List[str]:
        """List all available collections."""
        try:
            collections = self.client.get_collections().collections
            return [collection.name for collection in collections]
        except Exception as e:
            raise DocumentStoreError(f"Failed to list collections: {str(e)}")
    
    def index_exists(self, index_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(collection.name == index_name for collection in collections)
        except Exception as e:
            raise DocumentStoreError(f"Failed to check collection existence: {str(e)}")
    
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            # Get collection info
            collection_info = self.client.get_collection(collection_name=index_name)
            
            # Get metadata if available
            try:
                metadata_points = self.client.scroll(
                    collection_name=index_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="__is_metadata__", match=MatchValue(value=True))]
                    ),
                    limit=1
                )[0]
                
                if metadata_points:
                    stored_metadata = metadata_points[0].payload.get("metadata", {})
                else:
                    stored_metadata = {}
            except:
                stored_metadata = {}
            
            # Count documents (excluding metadata)
            count_result = self.client.count(
                collection_name=index_name,
                count_filter=Filter(
                    must_not=[FieldCondition(key="__is_metadata__", match=MatchValue(value=True))]
                )
            )
            
            return {
                "name": index_name,
                "status": collection_info.status.value,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "document_count": count_result.count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.value,
                **stored_metadata
            }
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to get collection info: {str(e)}")
    
    # Document CRUD Operations
    
    def add_document(
        self,
        document: Document,
        index_name: str,
        auto_embed: bool = True,
        overwrite: bool = False
    ) -> bool:
        """Add a document to the specified collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            # Check if document already exists
            if not overwrite and self.document_exists(document.id, index_name):
                raise DuplicateDocumentError(f"Document '{document.id}' already exists in collection '{index_name}'")
            
            # Auto-generate embedding if requested and not present
            if auto_embed and document.embedding is None and self.embedding_model:
                document = self.embedding_model.encode_document(document)
            
            # Validate embedding
            if document.embedding is None:
                raise DocumentStoreError("Document must have an embedding to be stored in Qdrant")
            
            # Prepare document payload
            payload = {
                "content": document.content,
                "metadata": document.metadata,
                "doc_id": document.id,
                "updated_at": datetime.now().isoformat()
            }
            
            # Create point
            point = PointStruct(
                id=document.id,
                vector=document.embedding,
                payload=payload
            )
            
            # Upsert point
            self.client.upsert(
                collection_name=index_name,
                points=[point]
            )
            
            return True
            
        except (IndexNotFoundError, DuplicateDocumentError):
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to add document '{document.id}': {str(e)}")
    
    def add_documents(
        self,
        documents: List[Document],
        index_name: str,
        auto_embed: bool = True,
        overwrite: bool = False,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Add multiple documents to the specified collection in batch."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            added_count = 0
            failed_count = 0
            errors = []
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_points = []
                
                for doc in batch:
                    try:
                        # Check for duplicates if not overwriting
                        if not overwrite and self.document_exists(doc.id, index_name):
                            failed_count += 1
                            errors.append(f"Document '{doc.id}' already exists")
                            continue
                        
                        # Auto-generate embedding if needed
                        if auto_embed and doc.embedding is None and self.embedding_model:
                            doc = self.embedding_model.encode_document(doc)
                        
                        if doc.embedding is None:
                            failed_count += 1
                            errors.append(f"Document '{doc.id}' has no embedding")
                            continue
                        
                        # Prepare point
                        payload = {
                            "content": doc.content,
                            "metadata": doc.metadata,
                            "doc_id": doc.id,
                            "updated_at": datetime.now().isoformat()
                        }
                        
                        point = PointStruct(
                            id=doc.id,
                            vector=doc.embedding,
                            payload=payload
                        )
                        batch_points.append(point)
                        
                    except Exception as e:
                        failed_count += 1
                        errors.append(f"Document '{doc.id}': {str(e)}")
                
                # Upsert batch
                if batch_points:
                    try:
                        self.client.upsert(
                            collection_name=index_name,
                            points=batch_points
                        )
                        added_count += len(batch_points)
                    except Exception as e:
                        failed_count += len(batch_points)
                        errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
            
            return {
                "added_count": added_count,
                "failed_count": failed_count,
                "total_count": len(documents),
                "errors": errors
            }
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to add documents: {str(e)}")
    
    def get_document(self, document_id: str, index_name: str) -> Document:
        """Retrieve a document by ID from the specified collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            # Retrieve point
            points = self.client.retrieve(
                collection_name=index_name,
                ids=[document_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                raise DocumentNotFoundError(f"Document '{document_id}' not found in collection '{index_name}'")
            
            point = points[0]
            
            # Convert point back to Document
            document = Document(
                id=point.payload["doc_id"],
                content=point.payload["content"],
                metadata=point.payload["metadata"],
                embedding=point.vector
            )
            
            return document
            
        except (IndexNotFoundError, DocumentNotFoundError):
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to retrieve document '{document_id}': {str(e)}")
    
    def get_documents(
        self,
        document_ids: List[str],
        index_name: str,
        include_missing: bool = False
    ) -> List[Document]:
        """Retrieve multiple documents by IDs from the specified collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            # Retrieve points
            points = self.client.retrieve(
                collection_name=index_name,
                ids=document_ids,
                with_payload=True,
                with_vectors=True
            )
            
            # Create lookup for found points
            found_points = {point.id: point for point in points}
            
            documents = []
            for doc_id in document_ids:
                if doc_id in found_points:
                    point = found_points[doc_id]
                    document = Document(
                        id=point.payload["doc_id"],
                        content=point.payload["content"],
                        metadata=point.payload["metadata"],
                        embedding=point.vector
                    )
                    documents.append(document)
                elif include_missing:
                    documents.append(None)
            
            return documents
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to retrieve documents: {str(e)}")
    
    def update_document(
        self,
        document: Document,
        index_name: str,
        auto_embed: bool = True
    ) -> bool:
        """Update an existing document in the specified collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            if not self.document_exists(document.id, index_name):
                raise DocumentNotFoundError(f"Document '{document.id}' not found in collection '{index_name}'")
            
            # Use add_document with overwrite=True
            return self.add_document(document, index_name, auto_embed, overwrite=True)
            
        except (IndexNotFoundError, DocumentNotFoundError):
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to update document '{document.id}': {str(e)}")
    
    def delete_document(self, document_id: str, index_name: str) -> bool:
        """Delete a document from the specified collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            if not self.document_exists(document_id, index_name):
                raise DocumentNotFoundError(f"Document '{document_id}' not found in collection '{index_name}'")
            
            # Delete point
            self.client.delete(
                collection_name=index_name,
                points_selector=[document_id]
            )
            
            return True
            
        except (IndexNotFoundError, DocumentNotFoundError):
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to delete document '{document_id}': {str(e)}")
    
    def delete_documents(
        self,
        document_ids: List[str],
        index_name: str,
        ignore_missing: bool = True
    ) -> Dict[str, Any]:
        """Delete multiple documents from the specified collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            deleted_count = 0
            failed_count = 0
            errors = []
            
            # Check which documents exist if not ignoring missing
            if not ignore_missing:
                existing_docs = self.get_documents(document_ids, index_name, include_missing=False)
                existing_ids = {doc.id for doc in existing_docs}
                
                for doc_id in document_ids:
                    if doc_id not in existing_ids:
                        failed_count += 1
                        errors.append(f"Document '{doc_id}' not found")
                        continue
            
            # Delete documents
            try:
                self.client.delete(
                    collection_name=index_name,
                    points_selector=document_ids
                )
                
                # Count successful deletions (approximate)
                deleted_count = len(document_ids) - failed_count
                
            except Exception as e:
                failed_count = len(document_ids)
                errors.append(f"Batch deletion failed: {str(e)}")
            
            return {
                "deleted_count": deleted_count,
                "failed_count": failed_count,
                "total_count": len(document_ids),
                "errors": errors
            }
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to delete documents: {str(e)}")
    
    def document_exists(self, document_id: str, index_name: str) -> bool:
        """Check if a document exists in the specified collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            points = self.client.retrieve(
                collection_name=index_name,
                ids=[document_id],
                with_payload=False,
                with_vectors=False
            )
            
            return len(points) > 0
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            return False
    
    def count_documents(
        self, 
        index_name: str, 
        filters: Optional[Dict[str, Any]] = None,
        return_embeddings: Optional[bool] = None
    ) -> int:
        """Count documents in the specified collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            # Build filter conditions
            filter_conditions = []
            
            # Exclude metadata points
            filter_conditions.append(
                FieldCondition(key="__is_metadata__", match=MatchValue(value=True))
            )
            
            # Apply custom filters
            if filters:
                for key, value in filters.items():
                    filter_conditions.append(
                        FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                    )
            
            # Note: return_embeddings filter not applicable in Qdrant 
            # (all stored documents have embeddings)
            
            count_filter = Filter(must_not=[filter_conditions[0]])
            if len(filter_conditions) > 1:
                count_filter.must = filter_conditions[1:]
            
            # Count documents
            count_result = self.client.count(
                collection_name=index_name,
                count_filter=count_filter
            )
            
            return count_result.count
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to count documents: {str(e)}")
    
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
        """Query documents in the specified collection using semantic similarity."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            # Validate and normalize query
            if isinstance(query, str):
                query = Query(content=query)
            
            # Generate query embedding if not present
            if query.embedding is None:
                if self.embedding_model is None:
                    raise DocumentStoreError("No embedding model available for query encoding")
                query = self.embedding_model.encode_query(query)
            
            # Build filter conditions
            filter_conditions = []
            
            # Exclude metadata points
            filter_conditions.append(
                FieldCondition(key="__is_metadata__", match=MatchValue(value=True))
            )
            
            # Apply custom filters
            if filters:
                for key, value in filters.items():
                    filter_conditions.append(
                        FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                    )
            
            search_filter = Filter(must_not=[filter_conditions[0]])
            if len(filter_conditions) > 1:
                search_filter.must = filter_conditions[1:]
            
            # Perform search
            search_results = self.client.search(
                collection_name=index_name,
                query_vector=query.embedding,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=similarity_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to RetrievalResult objects
            results = []
            for i, scored_point in enumerate(search_results):
                document = Document(
                    id=scored_point.payload["doc_id"],
                    content=scored_point.payload["content"],
                    metadata=scored_point.payload["metadata"],
                    embedding=None  # Not retrieved for performance
                )
                
                result = RetrievalResult(
                    document=document,
                    score=scored_point.score,
                    rank=i + 1,
                    explanation=f"Qdrant similarity score: {scored_point.score:.4f}"
                )
                results.append(result)
            
            return results
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to query documents: {str(e)}")
    
    def filter_documents(
        self,
        index_name: str,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        offset: int = 0,
        return_embeddings: Optional[bool] = None
    ) -> List[Document]:
        """Filter documents by metadata criteria."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            # Build filter conditions
            filter_conditions = []
            
            # Exclude metadata points
            filter_conditions.append(
                FieldCondition(key="__is_metadata__", match=MatchValue(value=True))
            )
            
            # Apply custom filters
            for key, value in filters.items():
                filter_conditions.append(
                    FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                )
            
            search_filter = Filter(must_not=[filter_conditions[0]])
            if len(filter_conditions) > 1:
                search_filter.must = filter_conditions[1:]
            
            # Note: return_embeddings filter not applicable in Qdrant
            
            # Scroll through results
            documents = []
            next_page_offset = None
            current_offset = 0
            collected = 0
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=index_name,
                    scroll_filter=search_filter,
                    limit=min(100, (limit or 1000) - collected) if limit else 100,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=return_embeddings is True
                )
                
                points, next_page_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    # Apply offset
                    if current_offset < offset:
                        current_offset += 1
                        continue
                    
                    document = Document(
                        id=point.payload["doc_id"],
                        content=point.payload["content"],
                        metadata=point.payload["metadata"],
                        embedding=point.vector if return_embeddings else None
                    )
                    documents.append(document)
                    collected += 1
                    
                    # Check limit
                    if limit and collected >= limit:
                        break
                
                # Break if we have enough or no more pages
                if (limit and collected >= limit) or next_page_offset is None:
                    break
            
            return documents
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to filter documents: {str(e)}")
    
    # Embedding Management Methods
    
    def update_embeddings(
        self,
        index_name: str,
        documents: Optional[List[Document]] = None,
        batch_size: int = 100,
        force: bool = False
    ) -> Dict[str, Any]:
        """Update embeddings for documents in the specified collection."""
        try:
            if not self.index_exists(index_name):
                raise IndexNotFoundError(f"Collection '{index_name}' does not exist")
            
            if self.embedding_model is None:
                raise DocumentStoreError("No embedding model available")
            
            updated_count = 0
            failed_count = 0
            errors = []
            
            # In Qdrant, all stored documents already have embeddings
            # This method would re-generate embeddings for specified documents
            
            if documents is None:
                # Get all documents from collection
                all_docs = self.filter_documents(index_name, {}, return_embeddings=True)
                target_docs = all_docs
            else:
                target_docs = documents
            
            # Process in batches
            for i in range(0, len(target_docs), batch_size):
                batch = target_docs[i:i + batch_size]
                
                try:
                    # Re-generate embeddings
                    updated_batch = []
                    for doc in batch:
                        try:
                            # Create new document without embedding for re-encoding
                            doc_for_encoding = Document(
                                id=doc.id,
                                content=doc.content,
                                metadata=doc.metadata,
                                embedding=None
                            )
                            updated_doc = self.embedding_model.encode_document(doc_for_encoding)
                            updated_batch.append(updated_doc)
                        except Exception as e:
                            failed_count += 1
                            errors.append(f"Failed to encode document {doc.id}: {str(e)}")
                    
                    # Update documents in Qdrant
                    if updated_batch:
                        result = self.add_documents(updated_batch, index_name, auto_embed=False, overwrite=True)
                        updated_count += result.get('added_count', 0)
                        failed_count += result.get('failed_count', 0)
                        errors.extend(result.get('errors', []))
                        
                except Exception as e:
                    failed_count += len(batch)
                    errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
            
            return {
                "updated_count": updated_count,
                "failed_count": failed_count,
                "total_candidates": len(target_docs),
                "errors": errors
            }
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to update embeddings: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of the Qdrant document store."""
        try:
            indices = self.list_indices()
            total_docs = sum(self.count_documents(idx) for idx in indices)
            return f"QdrantDocumentStore(name='{self.name}', url='{self.qdrant_url}', indices={len(indices)}, total_docs={total_docs})"
        except:
            return f"QdrantDocumentStore(name='{self.name}', url='{self.qdrant_url}')"
