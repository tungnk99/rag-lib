"""
Base class for chunkers.

This module provides the abstract base class for all chunkers in the RAG system.
Chunkers are responsible for dividing FileContent into smaller, manageable Document chunks
that can be indexed and retrieved effectively.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
from ..schemas.schema import Document, Element, ElementType, FileContent


class ChunkerError(Exception):
    """Base exception for chunker operations."""
    pass


class InvalidChunkSizeError(ChunkerError):
    """Raised when chunk size parameters are invalid."""
    pass


class BaseChunker(ABC):
    """
    Abstract base class for all chunkers.
    
    Chunkers take a FileContent object (containing elements) and split it into
    smaller Document chunks that can be effectively indexed and retrieved.
    
    Attributes:
        name (str): Name/identifier of the chunker
        config (Dict[str, Any]): Chunker-specific configuration parameters
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            name (str): Name/identifier of the chunker
            config (Optional[Dict[str, Any]]): Chunker-specific configuration parameters
        """
        self.name = name
        self.config = config or {}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate chunker configuration parameters."""
        # Default implementation does nothing
        # Concrete classes should override this for specific validation
        pass
    
    @abstractmethod
    def chunk_file_content(self, file_content: FileContent) -> List[Document]:
        """
        Split file content into document chunks.
        
        Args:
            file_content (FileContent): File content to be chunked
            
        Returns:
            List[Document]: List of document chunks created from the file content
            
        Raises:
            ChunkerError: If chunking fails
            ValueError: If file content is invalid
        """
        pass
    
    def chunk_file_contents(self, file_contents: List[FileContent]) -> List[Document]:
        """
        Split multiple file contents into document chunks.
        
        Args:
            file_contents (List[FileContent]): List of file contents to be chunked
            
        Returns:
            List[Document]: List of all document chunks created from the file contents
            
        Raises:
            ChunkerError: If chunking fails
        """
        all_chunks = []
        for file_content in file_contents:
            try:
                chunks = self.chunk_file_content(file_content)
                all_chunks.extend(chunks)
            except Exception as e:
                raise ChunkerError(f"Failed to chunk file {file_content.file_path}: {str(e)}")
        
        return all_chunks
    
    def _create_document_chunk(
        self,
        content: str,
        file_content: FileContent,
        chunk_index: int,
        element_metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Create a document chunk from content and file metadata.
        
        Args:
            content (str): Content of the chunk
            file_content (FileContent): Source file content
            chunk_index (int): Index of this chunk within the file
            element_metadata (Optional[Dict[str, Any]]): Metadata from source elements
            
        Returns:
            Document: Created document chunk object
        """
        if not content.strip():
            raise ValueError("Chunk content cannot be empty")
        
        # Generate unique document ID based on file path and chunk index
        import os
        file_name = os.path.splitext(os.path.basename(file_content.file_path))[0]
        chunk_id = f"{file_name}_chunk_{chunk_index}"
        
        # Prepare chunk metadata
        chunk_metadata = {
            "source_file": file_content.file_path,
            "chunk_method": self.name,
            "chunk_index": chunk_index,
            "total_elements_in_source": len(file_content.elements),
            "created_at": datetime.now().isoformat(),
            "doc_type": "chunk"
        }
        
        # Add element metadata if provided
        if element_metadata:
            chunk_metadata.update(element_metadata)
        
        # Add file metadata (with prefix to avoid conflicts)
        for key, value in file_content.metadata.items():
            if key not in chunk_metadata:  # Don't override existing keys
                chunk_metadata[f"file_{key}"] = value
        
        # Add processing time if available
        if file_content.processing_time is not None:
            chunk_metadata["file_processing_time"] = file_content.processing_time
        
        return Document(
            id=chunk_id,
            content=content.strip(),
            metadata=chunk_metadata
        )
    
    def _get_text_content_from_elements(self, elements: List[Element]) -> str:
        """
        Extract text content from a list of elements.
        
        Args:
            elements (List[Element]): List of elements to extract text from
            
        Returns:
            str: Combined text content
        """
        text_parts = []
        for element in elements:
            if element.type == ElementType.TEXT:
                text_parts.append(element.content)
            elif element.type == ElementType.TABLE:
                # For tables, we might want to include the content as text
                text_parts.append(element.content)
            # Skip IMAGE elements for text extraction
        
        return " ".join(text_parts).strip()
    
    def _merge_element_metadata(self, elements: List[Element]) -> Dict[str, Any]:
        """
        Merge metadata from multiple elements.
        
        Args:
            elements (List[Element]): List of elements to merge metadata from
            
        Returns:
            Dict[str, Any]: Merged metadata
        """
        merged_metadata = {}
        
        # Collect all unique keys
        all_keys = set()
        for element in elements:
            all_keys.update(element.metadata.keys())
        
        # Merge values for each key
        for key in all_keys:
            values = [element.metadata.get(key) for element in elements if key in element.metadata]
            
            # Remove duplicates while preserving order
            unique_values = []
            for value in values:
                if value not in unique_values:
                    unique_values.append(value)
            
            # Store single value or list depending on count
            if len(unique_values) == 1:
                merged_metadata[key] = unique_values[0]
            else:
                merged_metadata[key] = unique_values
        
        return merged_metadata
    
    def _create_single_document_from_file_content(self, file_content: FileContent) -> Document:
        """
        Create a single document from entire file content (no chunking).
        
        Args:
            file_content (FileContent): Source file content
            
        Returns:
            Document: Created document object
        """
        # Get all text content from elements
        text_content = file_content.get_text_content()
        
        if not text_content.strip():
            raise ValueError("File content has no extractable text")
        
        # Generate document ID based on file path
        import os
        file_name = os.path.splitext(os.path.basename(file_content.file_path))[0]
        doc_id = f"{file_name}_full_document"
        
        # Prepare metadata
        doc_metadata = {
            "source_file": file_content.file_path,
            "chunk_method": self.name,
            "chunk_index": 0,
            "total_elements_in_source": len(file_content.elements),
            "elements_count_by_type": file_content.get_elements_count_by_type(),
            "created_at": datetime.now().isoformat(),
            "doc_type": "full_document"
        }
        
        # Add file metadata
        for key, value in file_content.metadata.items():
            doc_metadata[f"file_{key}"] = value
        
        # Add processing time if available
        if file_content.processing_time is not None:
            doc_metadata["file_processing_time"] = file_content.processing_time
        
        return Document(
            id=doc_id,
            content=text_content,
            metadata=doc_metadata
        )
    
    def get_chunker_info(self) -> Dict[str, Any]:
        """
        Get information about the chunker.
        
        Returns:
            Dict[str, Any]: Chunker information including name, config, etc.
        """
        return {
            "name": self.name,
            "config": self.config,
            "type": self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """String representation of the chunker."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the chunker."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"
