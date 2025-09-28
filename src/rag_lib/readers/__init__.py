"""
Readers module for RAG (Retrieval-Augmented Generation) system.

This module provides document readers for various file formats including
PDF, DOCX, and TXT files. Each reader extracts content as structured elements
(text, images, tables) that can be further processed in the RAG pipeline.

Available readers:
    - BaseReader: Abstract base class for all readers
    - TxtReader: For plain text files (.txt)
    - PdfReader: For PDF documents (.pdf) 
    - DocxReader: For Microsoft Word documents (.docx)
    - XlsxReader: For Excel spreadsheet files (.xlsx, .xlsm)
    - MultiFormatReader: For handling multiple file formats automatically
"""

from ._base import BaseReader, MultiFormatReader
from .txt_reader import TxtReader
from .pdf_reader import PdfReader
from .docx_reader import DocxReader
from .xlsx_reader import XlsxReader

__all__ = [
    'BaseReader',
    'MultiFormatReader', 
    'TxtReader',
    'PdfReader',
    'DocxReader',
    'XlsxReader'
]


def create_multi_format_reader() -> MultiFormatReader:
    """
    Create a multi-format reader with all available readers registered.
    
    Returns:
        MultiFormatReader: Configured reader that can handle multiple formats
    """
    reader = MultiFormatReader()
    
    # Register all available readers
    reader.register_reader(TxtReader())
    reader.register_reader(PdfReader())
    reader.register_reader(DocxReader())
    reader.register_reader(XlsxReader())
    
    return reader


def get_reader_for_file(file_path: str) -> BaseReader:
    """
    Get an appropriate reader for the given file path.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        BaseReader: Appropriate reader for the file format
        
    Raises:
        ValueError: If no suitable reader is found
    """
    from pathlib import Path
    
    extension = Path(file_path).suffix.lower()
    
    if extension in TxtReader.SUPPORTED_EXTENSIONS:
        return TxtReader()
    elif extension in PdfReader.SUPPORTED_EXTENSIONS:
        return PdfReader()
    elif extension in DocxReader.SUPPORTED_EXTENSIONS:
        return DocxReader()
    elif extension in XlsxReader.SUPPORTED_EXTENSIONS:
        return XlsxReader()
    else:
        raise ValueError(f"No reader available for file format: {extension}")


def get_supported_formats() -> dict:
    """
    Get all supported file formats and their corresponding readers.
    
    Returns:
        dict: Mapping of file extensions to reader classes
    """
    return {
        '.txt': TxtReader,
        '.text': TxtReader,
        '.pdf': PdfReader,
        '.docx': DocxReader,
        '.xlsx': XlsxReader,
        '.xlsm': XlsxReader
    }
