"""
PDF file reader for reading PDF documents.

This module provides functionality to read PDF files and extract
their content as text, image, and table elements.

Dependencies:
    - PyPDF2 or pdfplumber for text extraction
    - Optional: pdfplumber for table extraction
    - Optional: pdf2image for image extraction
"""

from typing import List, Optional
from pathlib import Path
from ._base import BaseReader
from ..schemas.schema import Element, ElementType, create_element


class PdfReader(BaseReader):
    """
    Reader for PDF files (.pdf).
    
    This reader extracts text, tables, and optionally images from PDF files.
    It uses different libraries based on availability and configuration.
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf'}
    
    def __init__(self, 
                 extract_tables: bool = True,
                 extract_images: bool = False,
                 split_by_pages: bool = True,
                 **kwargs):
        """
        Initialize the PDF reader.
        
        Args:
            extract_tables (bool): Whether to extract tables from PDF
            extract_images (bool): Whether to extract images from PDF
            split_by_pages (bool): Whether to split content by pages
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.split_by_pages = split_by_pages
        
        # Check for required dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import PyPDF2
            self.pdf_library = 'PyPDF2'
        except ImportError:
            try:
                import pdfplumber
                self.pdf_library = 'pdfplumber'
            except ImportError:
                raise ImportError(
                    "PDF reading requires either PyPDF2 or pdfplumber. "
                    "Install with: pip install PyPDF2 or pip install pdfplumber"
                )
        
        if self.extract_tables:
            try:
                import pdfplumber
                self.table_extractor = 'pdfplumber'
            except ImportError:
                print("Warning: pdfplumber not found. Table extraction disabled.")
                self.extract_tables = False
        
        if self.extract_images:
            try:
                import pdf2image
                self.image_extractor = 'pdf2image'
            except ImportError:
                print("Warning: pdf2image not found. Image extraction disabled.")
                self.extract_images = False
    
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
    
    def read(self, file_path: str) -> List[Element]:
        """
        Read a PDF file and extract its content.
        
        Args:
            file_path (str): Path to the PDF file to read
            
        Returns:
            List[Element]: List of extracted elements (text, tables, images)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other reading errors
        """
        elements = []
        
        if self.pdf_library == 'PyPDF2':
            elements.extend(self._read_with_pypdf2(file_path))
        elif self.pdf_library == 'pdfplumber':
            elements.extend(self._read_with_pdfplumber(file_path))
        
        if self.extract_images:
            elements.extend(self._extract_images(file_path))
        
        return elements
    
    def _read_with_pypdf2(self, file_path: str) -> List[Element]:
        """Read PDF using PyPDF2 library."""
        try:
            import PyPDF2
            
            elements = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if self.split_by_pages:
                    # Extract text page by page
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            metadata = {
                                'page_number': page_num + 1,
                                'total_pages': len(pdf_reader.pages),
                                'extraction_method': 'PyPDF2'
                            }
                            element = create_element(
                                element_type=ElementType.TEXT,
                                content=text,
                                metadata=metadata
                            )
                            elements.append(element)
                else:
                    # Extract all text as single element
                    all_text = ""
                    for page in pdf_reader.pages:
                        all_text += page.extract_text() + "\n"
                    
                    if all_text.strip():
                        metadata = {
                            'total_pages': len(pdf_reader.pages),
                            'extraction_method': 'PyPDF2'
                        }
                        element = create_element(
                            element_type=ElementType.TEXT,
                            content=all_text,
                            metadata=metadata
                        )
                        elements.append(element)
            
            return elements
            
        except Exception as e:
            raise Exception(f"Error reading PDF with PyPDF2: {str(e)}")
    
    def _read_with_pdfplumber(self, file_path: str) -> List[Element]:
        """Read PDF using pdfplumber library."""
        try:
            import pdfplumber
            
            elements = []
            with pdfplumber.open(file_path) as pdf:
                
                if self.split_by_pages:
                    # Extract content page by page
                    for page_num, page in enumerate(pdf.pages):
                        # Extract text
                        text = page.extract_text()
                        if text and text.strip():
                            metadata = {
                                'page_number': page_num + 1,
                                'total_pages': len(pdf.pages),
                                'extraction_method': 'pdfplumber'
                            }
                            element = create_element(
                                element_type=ElementType.TEXT,
                                content=text,
                                metadata=metadata
                            )
                            elements.append(element)
                        
                        # Extract tables if enabled
                        if self.extract_tables:
                            tables = page.extract_tables()
                            for table_num, table in enumerate(tables):
                                if table:
                                    # Convert table to string representation
                                    table_text = self._table_to_text(table)
                                    metadata = {
                                        'page_number': page_num + 1,
                                        'table_number': table_num + 1,
                                        'total_pages': len(pdf.pages),
                                        'extraction_method': 'pdfplumber'
                                    }
                                    element = create_element(
                                        element_type=ElementType.TABLE,
                                        content=table_text,
                                        metadata=metadata
                                    )
                                    elements.append(element)
                else:
                    # Extract all content as combined elements
                    all_text = ""
                    all_tables = []
                    
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            all_text += text + "\n"
                        
                        if self.extract_tables:
                            tables = page.extract_tables()
                            all_tables.extend(tables)
                    
                    # Add text element
                    if all_text.strip():
                        metadata = {
                            'total_pages': len(pdf.pages),
                            'extraction_method': 'pdfplumber'
                        }
                        element = create_element(
                            element_type=ElementType.TEXT,
                            content=all_text,
                            metadata=metadata
                        )
                        elements.append(element)
                    
                    # Add table elements
                    for table_num, table in enumerate(all_tables):
                        if table:
                            table_text = self._table_to_text(table)
                            metadata = {
                                'table_number': table_num + 1,
                                'total_tables': len(all_tables),
                                'extraction_method': 'pdfplumber'
                            }
                            element = create_element(
                                element_type=ElementType.TABLE,
                                content=table_text,
                                metadata=metadata
                            )
                            elements.append(element)
            
            return elements
            
        except Exception as e:
            raise Exception(f"Error reading PDF with pdfplumber: {str(e)}")
    
    def _extract_images(self, file_path: str) -> List[Element]:
        """Extract images from PDF using pdf2image."""
        try:
            import pdf2image
            from PIL import Image
            import base64
            from io import BytesIO
            
            elements = []
            images = pdf2image.convert_from_path(file_path)
            
            for img_num, image in enumerate(images):
                # Convert image to base64 string
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                metadata = {
                    'image_number': img_num + 1,
                    'total_images': len(images),
                    'format': 'PNG',
                    'extraction_method': 'pdf2image',
                    'encoding': 'base64'
                }
                
                element = create_element(
                    element_type=ElementType.IMAGE,
                    content=img_base64,
                    metadata=metadata
                )
                elements.append(element)
            
            return elements
            
        except Exception as e:
            print(f"Warning: Error extracting images from PDF: {str(e)}")
            return []
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to text representation."""
        if not table:
            return ""
        
        # Simple CSV-like format
        lines = []
        for row in table:
            # Handle None values and convert to strings
            clean_row = [str(cell) if cell is not None else "" for cell in row]
            lines.append(" | ".join(clean_row))
        
        return "\n".join(lines)
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions (without dots)
        """
        return [ext.lstrip('.') for ext in self.SUPPORTED_EXTENSIONS]
