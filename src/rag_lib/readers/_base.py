"""
Base reader class for document reading functionality.

This module provides the abstract base class for all document readers in the RAG system.
Each reader is responsible for extracting content from specific file formats and
returning a FileContent object containing elements (text, image, table) and metadata.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..schemas.schema import Element, FileContent


class BaseReader(ABC):
    """
    Abstract base class for all document readers.
    
    This class defines the interface that all concrete readers must implement.
    Each reader should be able to read a specific file format and extract
    its content as a FileContent object.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the reader with optional configuration.
        
        Args:
            **kwargs: Additional configuration parameters for the reader
        """
        self.config = kwargs
    
    @abstractmethod
    def read(self, file_path: str) -> FileContent:
        """
        Read a file and extract its content as a FileContent object.
        
        Args:
            file_path (str): Path to the file to read
            
        Returns:
            FileContent: File content object containing elements and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported or invalid
            Exception: For other reading errors
        """
        pass
    
    @abstractmethod
    def supports_format(self, file_path: str) -> bool:
        """
        Check if this reader supports the given file format.
        
        Args:
            file_path (str): Path to the file to check
            
        Returns:
            bool: True if this reader can handle the file format, False otherwise
        """
        pass
    
    def validate_file(self, file_path: str) -> None:
        """
        Validate that the file exists and is readable.
        
        Args:
            file_path (str): Path to the file to validate
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file is not readable
            ValueError: If the file path is invalid
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if not path.stat().st_size > 0:
            raise ValueError(f"File is empty: {file_path}")
    
    def get_file_extension(self, file_path: str) -> str:
        """
        Get the file extension from the file path.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: File extension (lowercase, without the dot)
        """
        return Path(file_path).suffix.lower().lstrip('.')
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic metadata about the file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Dict[str, Any]: File metadata (size, modified_time, etc.)
        """
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "file_path": str(path.absolute()),
            "file_name": path.name,
            "file_size": stat.st_size,
            "modified_time": stat.st_mtime,
            "extension": self.get_file_extension(file_path)
        }
    
    def create_file_content(self, file_path: str, elements: List[Element], 
                           processing_time: Optional[float] = None) -> FileContent:
        """
        Create a FileContent object with file metadata.
        
        Args:
            file_path (str): Path to the file
            elements (List[Element]): List of extracted elements
            processing_time (Optional[float]): Time taken to process the file
            
        Returns:
            FileContent: File content object with metadata
        """
        file_metadata = self.get_file_metadata(file_path)
        
        return FileContent(
            file_path=file_path,
            elements=elements,
            metadata=file_metadata,
            processing_time=processing_time
        )
    
    def read_with_validation(self, file_path: str) -> FileContent:
        """
        Read a file with validation and return FileContent.
        
        Args:
            file_path (str): Path to the file to read
            
        Returns:
            FileContent: File content object containing elements and metadata
        """
        # Validate file first
        self.validate_file(file_path)
        
        # Check if this reader supports the file format
        if not self.supports_format(file_path):
            raise ValueError(f"File format not supported by {self.__class__.__name__}: {file_path}")
        
        # Read the file and return FileContent
        return self.read(file_path)
    
    def __str__(self) -> str:
        """String representation of the reader."""
        return f"{self.__class__.__name__}({self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the reader."""
        return self.__str__()


class MultiFormatReader:
    """
    A reader that can handle multiple file formats by delegating to specific readers.
    Returns FileContent objects containing extracted elements and metadata.
    """
    
    def __init__(self):
        """Initialize the multi-format reader."""
        self.readers: List[BaseReader] = []
    
    def register_reader(self, reader: BaseReader) -> None:
        """
        Register a reader for a specific file format.
        
        Args:
            reader (BaseReader): The reader to register
        """
        if not isinstance(reader, BaseReader):
            raise ValueError("Reader must be an instance of BaseReader")
        
        self.readers.append(reader)
    
    def get_suitable_reader(self, file_path: str) -> Optional[BaseReader]:
        """
        Find a suitable reader for the given file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Optional[BaseReader]: The first reader that supports the file format, or None
        """
        for reader in self.readers:
            if reader.supports_format(file_path):
                return reader
        return None
    
    def read(self, file_path: str) -> FileContent:
        """
        Read a file using an appropriate reader.
        
        Args:
            file_path (str): Path to the file to read
            
        Returns:
            FileContent: File content object containing elements and metadata
            
        Raises:
            ValueError: If no suitable reader is found for the file format
        """
        reader = self.get_suitable_reader(file_path)
        if reader is None:
            extension = Path(file_path).suffix.lower()
            raise ValueError(f"No reader available for file format: {extension}")
        
        return reader.read_with_validation(file_path)
    
    def list_supported_formats(self) -> List[str]:
        """
        Get a list of all supported file formats.
        
        Returns:
            List[str]: List of supported file extensions
        """
        formats = set()
        for reader in self.readers:
            # This would need to be implemented by each reader
            # For now, we'll return a placeholder
            pass
        return list(formats)
