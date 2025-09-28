"""
Schema definitions for RAG (Retrieval-Augmented Generation) system.

This module contains the core data structures for Query and Document objects
used throughout the RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
import uuid
import os


class ElementType(Enum):
    """Enum for different types of document elements."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class Element:
    """
    Represents an element extracted from a document.
    
    Attributes:
        type (ElementType): The type of element (text, image, table)
        content (str): The content of the element (text content, image path/base64, table as string/html)
        metadata (Dict[str, Any]): Additional metadata (page_number, coordinates, format, etc.)
    """
    type: ElementType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate element after initialization."""
        if not self.content.strip():
            raise ValueError("Element content cannot be empty")
        
        # Set default metadata if not provided
        if "page_number" not in self.metadata:
            self.metadata["page_number"] = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dictionary representation."""
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Element':
        """Create Element from dictionary representation."""
        data = data.copy()
        data["type"] = ElementType(data["type"])
        return cls(**data)


@dataclass
class FileContent:
    """
    Represents the content extracted from a file by a reader.
    
    Attributes:
        file_path (str): Path to the source file
        elements (List[Element]): List of elements extracted from the file
        metadata (Dict[str, Any]): File-level metadata (file_size, file_type, processed_at, etc.)
        processing_time (Optional[float]): Time taken to process the file in seconds
    """
    file_path: str
    elements: List[Element] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        """Validate file content after initialization."""
        if not self.file_path.strip():
            raise ValueError("File path cannot be empty")
        
        if self.processing_time is not None and self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")
        
        # Set default metadata if not provided
        if "processed_at" not in self.metadata:
            self.metadata["processed_at"] = datetime.now().isoformat()
        
        if "file_type" not in self.metadata:
            # Try to infer file type from extension
            _, ext = os.path.splitext(self.file_path)
            self.metadata["file_type"] = ext.lower() if ext else "unknown"
    
    def get_elements_by_type(self, element_type: ElementType) -> List[Element]:
        """Get all elements of a specific type."""
        return [element for element in self.elements if element.type == element_type]
    
    def get_text_content(self) -> str:
        """Get all text content concatenated."""
        text_elements = self.get_elements_by_type(ElementType.TEXT)
        return "\n".join(element.content for element in text_elements)
    
    def get_total_elements_count(self) -> int:
        """Get total number of elements."""
        return len(self.elements)
    
    def get_elements_count_by_type(self) -> Dict[str, int]:
        """Get count of elements by type."""
        count = {}
        for element in self.elements:
            element_type = element.type.value
            count[element_type] = count.get(element_type, 0) + 1
        return count
    
    def add_element(self, element: Element) -> None:
        """Add an element to the file content."""
        if not isinstance(element, Element):
            raise TypeError("Must be an Element instance")
        self.elements.append(element)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert file content to dictionary representation."""
        return {
            "file_path": self.file_path,
            "elements": [element.to_dict() for element in self.elements],
            "metadata": self.metadata,
            "processing_time": self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileContent':
        """Create FileContent from dictionary representation."""
        data = data.copy()
        data["elements"] = [Element.from_dict(element_data) for element_data in data.get("elements", [])]
        return cls(**data)


@dataclass
class Query:
    """
    Represents a user query in the RAG system.
    
    Attributes:
        content (str): The main query text content
        metadata (Dict[str, Any]): Additional metadata for the query (can include query_id, timestamp, intent, max_results, filters, etc.)
        embedding (Optional[List[float]]): Query embedding vector if computed
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate and set default metadata after initialization."""
        if not self.content.strip():
            raise ValueError("Query content cannot be empty")
        
        # Set default metadata if not provided
        if "query_id" not in self.metadata:
            self.metadata["query_id"] = str(uuid.uuid4())
        
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()
        
        if "max_results" not in self.metadata:
            self.metadata["max_results"] = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary representation."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """Create Query from dictionary representation."""
        return cls(**data)


@dataclass
class Document:
    """
    Represents a document in the RAG system.
    
    Attributes:
        id (str): Unique identifier for the document
        content (str): The main document content/text
        metadata (Dict[str, Any]): Additional metadata about the document (can include title, source, doc_type, created_at, updated_at, chunk_ids, quality_score, etc.)
        embedding (Optional[List[float]]): Document embedding vector if computed
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate document after initialization."""
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")
        
        if not self.id.strip():
            raise ValueError("Document ID cannot be empty")
        
        # Set default metadata if not provided
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
        
        if "updated_at" not in self.metadata:
            self.metadata["updated_at"] = datetime.now().isoformat()
        
        if "doc_type" not in self.metadata:
            self.metadata["doc_type"] = "text"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from dictionary representation."""
        return cls(**data)


@dataclass
class RetrievalResult:
    """
    Represents a retrieval result containing a document and its relevance score.
    
    Attributes:
        document (Document): The retrieved document
        score (float): Relevance score (0-1, higher is more relevant)
        rank (int): Rank in the retrieval results (1-based)
        explanation (Optional[str]): Explanation of why this document was retrieved
    """
    document: Document
    score: float
    rank: int = 0
    explanation: Optional[str] = None
    
    def __post_init__(self):
        """Validate retrieval result after initialization."""
        if not (0 <= self.score <= 1):
            raise ValueError("Score must be between 0 and 1")
        
        if self.rank < 0:
            raise ValueError("Rank must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval result to dictionary representation."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "explanation": self.explanation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalResult':
        """Create RetrievalResult from dictionary representation."""
        data = data.copy()
        data["document"] = Document.from_dict(data["document"])
        return cls(**data)


@dataclass
class RAGResponse:
    """
    Represents the complete response from a RAG system.
    
    Attributes:
        query (Query): The original query
        generated_text (str): The generated response text
        retrieved_documents (List[RetrievalResult]): List of retrieved documents with scores
        generation_metadata (Dict[str, Any]): Metadata about the generation process
        total_time (Optional[float]): Total processing time in seconds
        retrieval_time (Optional[float]): Time spent on retrieval in seconds
        generation_time (Optional[float]): Time spent on generation in seconds
    """
    query: Query
    generated_text: str
    retrieved_documents: List[RetrievalResult] = field(default_factory=list)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    total_time: Optional[float] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    
    def __post_init__(self):
        """Validate RAG response after initialization."""
        if not self.generated_text.strip():
            raise ValueError("Generated text cannot be empty")
        
        # Validate time fields
        for time_field in [self.total_time, self.retrieval_time, self.generation_time]:
            if time_field is not None and time_field < 0:
                raise ValueError("Time values must be non-negative")
    
    def get_top_documents(self, n: int = 3) -> List[RetrievalResult]:
        """Get top N retrieved documents by score."""
        return sorted(self.retrieved_documents, key=lambda x: x.score, reverse=True)[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert RAG response to dictionary representation."""
        return {
            "query": self.query.to_dict(),
            "generated_text": self.generated_text,
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "generation_metadata": self.generation_metadata,
            "total_time": self.total_time,
            "retrieval_time": self.retrieval_time,
            "generation_time": self.generation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGResponse':
        """Create RAGResponse from dictionary representation."""
        data = data.copy()
        data["query"] = Query.from_dict(data["query"])
        data["retrieved_documents"] = [RetrievalResult.from_dict(doc) for doc in data["retrieved_documents"]]
        return cls(**data)


# Utility functions for schema validation and creation

def create_query(content: str, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None) -> Query:
    """
    Convenience function to create a Query object.
    
    Args:
        content (str): Query content
        metadata (Optional[Dict[str, Any]]): Query metadata
        embedding (Optional[List[float]]): Query embedding vector
    
    Returns:
        Query: Created query object
    """
    return Query(content=content, metadata=metadata or {}, embedding=embedding)


def create_document(doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None) -> Document:
    """
    Convenience function to create a Document object.
    
    Args:
        doc_id (str): Document ID
        content (str): Document content
        metadata (Optional[Dict[str, Any]]): Document metadata
        embedding (Optional[List[float]]): Document embedding vector
    
    Returns:
        Document: Created document object
    """
    return Document(id=doc_id, content=content, metadata=metadata or {}, embedding=embedding)


def validate_embedding(embedding: List[float], expected_dim: Optional[int] = None) -> bool:
    """
    Validate embedding vector.
    
    Args:
        embedding (List[float]): Embedding vector
        expected_dim (Optional[int]): Expected dimension
    
    Returns:
        bool: True if valid
    """
    if not isinstance(embedding, list):
        return False
    
    if not all(isinstance(x, (int, float)) for x in embedding):
        return False
    
    if expected_dim is not None and len(embedding) != expected_dim:
        return False
    
    return True


def batch_create_documents(documents_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Create multiple documents from a list of dictionaries.
    
    Args:
        documents_data (List[Dict[str, Any]]): List of document data dictionaries
    
    Returns:
        List[Document]: List of created document objects
    """
    return [Document.from_dict(doc_data) for doc_data in documents_data]


def create_element(element_type: ElementType, content: str, metadata: Optional[Dict[str, Any]] = None) -> Element:
    """
    Convenience function to create an Element object.
    
    Args:
        element_type (ElementType): Type of element
        content (str): Element content
        metadata (Optional[Dict[str, Any]]): Element metadata
    
    Returns:
        Element: Created element object
    """
    return Element(type=element_type, content=content, metadata=metadata or {})


def batch_create_elements(elements_data: List[Dict[str, Any]]) -> List[Element]:
    """
    Create multiple elements from a list of dictionaries.
    
    Args:
        elements_data (List[Dict[str, Any]]): List of element data dictionaries
    
    Returns:
        List[Element]: List of created element objects
    """
    return [Element.from_dict(element_data) for element_data in elements_data]


def create_file_content(file_path: str, elements: Optional[List[Element]] = None, 
                       metadata: Optional[Dict[str, Any]] = None, 
                       processing_time: Optional[float] = None) -> FileContent:
    """
    Convenience function to create a FileContent object.
    
    Args:
        file_path (str): Path to the source file
        elements (Optional[List[Element]]): List of elements extracted from the file
        metadata (Optional[Dict[str, Any]]): File-level metadata
        processing_time (Optional[float]): Time taken to process the file
    
    Returns:
        FileContent: Created file content object
    """
    return FileContent(
        file_path=file_path,
        elements=elements or [],
        metadata=metadata or {},
        processing_time=processing_time
    )


def batch_create_file_contents(file_contents_data: List[Dict[str, Any]]) -> List[FileContent]:
    """
    Create multiple file contents from a list of dictionaries.
    
    Args:
        file_contents_data (List[Dict[str, Any]]): List of file content data dictionaries
    
    Returns:
        List[FileContent]: List of created file content objects
    """
    return [FileContent.from_dict(file_data) for file_data in file_contents_data]
