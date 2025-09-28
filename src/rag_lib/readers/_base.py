"""
Base reader class for document reading functionality.

This module provides the abstract base class for all document readers in the RAG system.
Each reader is responsible for extracting content from specific file formats and
returning a list of elements (text, image, table).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
from ..schemas.schema import Element


class BaseReader(ABC):
    """
    Abstract base class for all document readers.
    
    This class defines the interface that all concrete readers must implement.
    Each reader should be able to read a specific file format and extract
    its content as a list of elements.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the reader with optional configuration.
        
        Args:
            **kwargs: Additional configuration parameters for the reader
        """
        self.config = kwargs
    
    @abstractmethod
    def read(self, file_path: str) -> List[Element]:
        """
        Read a file and extract its content as a list of elements.
        
        Args:
            file_path (str): Path to the file to read
            
        Returns:
            List[Element]: List of extracted elements (text, image, table)
            
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
    
    def read_with_metadata(self, file_path: str, include_file_metadata: bool = True) -> List[Element]:
        """
        Read a file and optionally include file metadata in each element.
        
        Args:
            file_path (str): Path to the file to read
            include_file_metadata (bool): Whether to include file metadata in elements
            
        Returns:
            List[Element]: List of extracted elements with optional file metadata
        """
        # Validate file first
        self.validate_file(file_path)
        
        # Check if this reader supports the file format
        if not self.supports_format(file_path):
            raise ValueError(f"File format not supported by {self.__class__.__name__}: {file_path}")
        
        # Read the file
        elements = self.read(file_path)
        
        # Add file metadata to each element if requested
        if include_file_metadata:
            file_metadata = self.get_file_metadata(file_path)
            for element in elements:
                element.metadata.update(file_metadata)
        
        return elements
    
    def __str__(self) -> str:
        """String representation of the reader."""
        return f"{self.__class__.__name__}({self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the reader."""
        return self.__str__()


class MultiFormatReader:
    """
    A reader that can handle multiple file formats by delegating to specific readers.
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
    
    def read(self, file_path: str, include_file_metadata: bool = True) -> List[Element]:
        """
        Read a file using an appropriate reader.
        
        Args:
            file_path (str): Path to the file to read
            include_file_metadata (bool): Whether to include file metadata in elements
            
        Returns:
            List[Element]: List of extracted elements
            
        Raises:
            ValueError: If no suitable reader is found for the file format
        """
        reader = self.get_suitable_reader(file_path)
        if reader is None:
            extension = Path(file_path).suffix.lower()
            raise ValueError(f"No reader available for file format: {extension}")
        
        return reader.read_with_metadata(file_path, include_file_metadata)
    
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
