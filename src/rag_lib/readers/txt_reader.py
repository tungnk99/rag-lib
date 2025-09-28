"""
Text file reader for reading plain text files.

This module provides functionality to read text files (.txt) and extract
their content as text elements.
"""

from typing import List
from pathlib import Path
import time
from ._base import BaseReader
from ..schemas.schema import Element, ElementType, create_element, FileContent


class TxtReader(BaseReader):
    """
    Reader for plain text files (.txt).
    
    This reader extracts text content from plain text files and returns
    it as a list of text elements. It can split content by lines or
    keep it as a single element.
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.text'}
    
    def __init__(self, split_by_lines: bool = False, encoding: str = 'utf-8', **kwargs):
        """
        Initialize the TXT reader.
        
        Args:
            split_by_lines (bool): Whether to split content into separate elements by lines
            encoding (str): Text encoding to use when reading files
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.split_by_lines = split_by_lines
        self.encoding = encoding
    
    def supports_format(self, file_path: str) -> bool:
        """
        Check if this reader supports the given file format.
        
        Args:
            file_path (str): Path to the file to check
            
        Returns:
            bool: True if the file has a supported extension
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS
    
    def read(self, file_path: str) -> FileContent:
        """
        Read a text file and extract its content.
        
        Args:
            file_path (str): Path to the text file to read
            
        Returns:
            FileContent: File content object containing extracted elements and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnicodeDecodeError: If the file cannot be decoded with the specified encoding
            Exception: For other reading errors
        """
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as file:
                content = file.read()
            
            elements = []
            
            if not content.strip():
                # Even for empty files, create FileContent with empty elements
                processing_time = time.time() - start_time
                txt_metadata = {
                    'encoding': self.encoding,
                    'split_by_lines': self.split_by_lines,
                    'total_elements': 0,
                    'char_count': 0,
                    'word_count': 0,
                    'line_count': 0
                }
                return self.create_file_content(
                    file_path=file_path,
                    elements=elements,
                    processing_time=processing_time
                )
            
            if self.split_by_lines:
                # Split content by lines and create separate elements
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:  # Skip empty lines
                        metadata = {
                            'line_number': i + 1,
                            'total_lines': len(lines)
                        }
                        element = create_element(
                            element_type=ElementType.TEXT,
                            content=line,
                            metadata=metadata
                        )
                        elements.append(element)
            else:
                # Keep entire content as single element
                metadata = {
                    'line_count': len(content.split('\n')),
                    'char_count': len(content),
                    'word_count': len(content.split())
                }
                element = create_element(
                    element_type=ElementType.TEXT,
                    content=content,
                    metadata=metadata
                )
                elements.append(element)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create metadata
            txt_metadata = {
                'encoding': self.encoding,
                'split_by_lines': self.split_by_lines,
                'total_elements': len(elements),
                'char_count': len(content),
                'word_count': len(content.split()),
                'line_count': len(content.split('\n'))
            }
            
            # Create and return FileContent
            return self.create_file_content(
                file_path=file_path,
                elements=elements,
                processing_time=processing_time
            )
            
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                f"Cannot decode file {file_path} with encoding {self.encoding}: {str(e)}"
            )
        except Exception as e:
            raise Exception(f"Error reading text file {file_path}: {str(e)}")
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions (without dots)
        """
        return [ext.lstrip('.') for ext in self.SUPPORTED_EXTENSIONS]
