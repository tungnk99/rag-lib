"""
DOCX file reader for reading Microsoft Word documents.

This module provides functionality to read DOCX files and extract
their content as text, image, and table elements.

Dependencies:
    - python-docx for document parsing
    - Optional: Pillow for image handling
"""

from typing import List
from pathlib import Path
from ._base import BaseReader
from ..schemas.schema import Element, ElementType, create_element


class DocxReader(BaseReader):
    """
    Reader for Microsoft Word documents (.docx).
    
    This reader extracts text, tables, and optionally images from DOCX files.
    """
    
    SUPPORTED_EXTENSIONS = {'.docx'}
    
    def __init__(self, 
                 extract_tables: bool = True,
                 extract_images: bool = False,
                 split_by_paragraphs: bool = False,
                 **kwargs):
        """
        Initialize the DOCX reader.
        
        Args:
            extract_tables (bool): Whether to extract tables from document
            extract_images (bool): Whether to extract images from document
            split_by_paragraphs (bool): Whether to split content by paragraphs
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.split_by_paragraphs = split_by_paragraphs
        
        # Check for required dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import docx
            self.docx_available = True
        except ImportError:
            raise ImportError(
                "DOCX reading requires python-docx. "
                "Install with: pip install python-docx"
            )
        
        if self.extract_images:
            try:
                from PIL import Image
                self.image_support = True
            except ImportError:
                print("Warning: Pillow not found. Image extraction may be limited.")
                self.image_support = False
    
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
        Read a DOCX file and extract its content.
        
        Args:
            file_path (str): Path to the DOCX file to read
            
        Returns:
            List[Element]: List of extracted elements (text, tables, images)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other reading errors
        """
        try:
            from docx import Document
            
            elements = []
            doc = Document(file_path)
            
            # Extract paragraphs (text content)
            if self.split_by_paragraphs:
                elements.extend(self._extract_paragraphs(doc))
            else:
                elements.extend(self._extract_full_text(doc))
            
            # Extract tables
            if self.extract_tables:
                elements.extend(self._extract_tables(doc))
            
            # Extract images
            if self.extract_images:
                elements.extend(self._extract_images(doc))
            
            return elements
            
        except Exception as e:
            raise Exception(f"Error reading DOCX file {file_path}: {str(e)}")
    
    def _extract_paragraphs(self, doc) -> List[Element]:
        """Extract text content split by paragraphs."""
        elements = []
        
        for para_num, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if text:
                metadata = {
                    'paragraph_number': para_num + 1,
                    'total_paragraphs': len(doc.paragraphs),
                    'extraction_method': 'python-docx',
                    'element_type': 'paragraph'
                }
                
                # Add style information if available
                if paragraph.style:
                    metadata['style'] = paragraph.style.name
                
                element = create_element(
                    element_type=ElementType.TEXT,
                    content=text,
                    metadata=metadata
                )
                elements.append(element)
        
        return elements
    
    def _extract_full_text(self, doc) -> List[Element]:
        """Extract all text content as single element."""
        full_text = ""
        paragraph_count = 0
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                full_text += text + "\n"
                paragraph_count += 1
        
        if full_text.strip():
            metadata = {
                'total_paragraphs': paragraph_count,
                'extraction_method': 'python-docx',
                'element_type': 'full_document',
                'char_count': len(full_text),
                'word_count': len(full_text.split())
            }
            
            element = create_element(
                element_type=ElementType.TEXT,
                content=full_text,
                metadata=metadata
            )
            return [element]
        
        return []
    
    def _extract_tables(self, doc) -> List[Element]:
        """Extract tables from the document."""
        elements = []
        
        for table_num, table in enumerate(doc.tables):
            table_data = []
            
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append(cell.text.strip())
                table_data.append(row_data)
            
            if table_data:
                # Convert table to text representation
                table_text = self._table_to_text(table_data)
                
                metadata = {
                    'table_number': table_num + 1,
                    'total_tables': len(doc.tables),
                    'rows': len(table_data),
                    'columns': len(table_data[0]) if table_data else 0,
                    'extraction_method': 'python-docx'
                }
                
                element = create_element(
                    element_type=ElementType.TABLE,
                    content=table_text,
                    metadata=metadata
                )
                elements.append(element)
        
        return elements
    
    def _extract_images(self, doc) -> List[Element]:
        """Extract images from the document."""
        elements = []
        
        try:
            import base64
            from io import BytesIO
            
            # Get document relationships to find images
            rels = doc.part.rels
            image_count = 0
            
            for rel in rels:
                if "image" in rels[rel].target_ref:
                    image_count += 1
                    
                    try:
                        # Get image data
                        image_part = rels[rel].target_part
                        image_data = image_part.blob
                        
                        # Convert to base64
                        img_base64 = base64.b64encode(image_data).decode('utf-8')
                        
                        # Determine image format from content type
                        content_type = image_part.content_type
                        image_format = content_type.split('/')[-1].upper()
                        
                        metadata = {
                            'image_number': image_count,
                            'format': image_format,
                            'extraction_method': 'python-docx',
                            'encoding': 'base64',
                            'content_type': content_type
                        }
                        
                        element = create_element(
                            element_type=ElementType.IMAGE,
                            content=img_base64,
                            metadata=metadata
                        )
                        elements.append(element)
                        
                    except Exception as e:
                        print(f"Warning: Could not extract image {image_count}: {str(e)}")
            
        except Exception as e:
            print(f"Warning: Error extracting images from DOCX: {str(e)}")
        
        return elements
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to text representation."""
        if not table_data:
            return ""
        
        # Simple CSV-like format
        lines = []
        for row in table_data:
            lines.append(" | ".join(row))
        
        return "\n".join(lines)
    
    def get_document_info(self, file_path: str) -> dict:
        """
        Get document properties and metadata.
        
        Args:
            file_path (str): Path to the DOCX file
            
        Returns:
            dict: Document properties
        """
        try:
            from docx import Document
            
            doc = Document(file_path)
            core_props = doc.core_properties
            
            return {
                'title': core_props.title,
                'author': core_props.author,
                'subject': core_props.subject,
                'created': core_props.created,
                'modified': core_props.modified,
                'category': core_props.category,
                'comments': core_props.comments,
                'total_paragraphs': len(doc.paragraphs),
                'total_tables': len(doc.tables)
            }
            
        except Exception as e:
            print(f"Warning: Could not extract document properties: {str(e)}")
            return {}
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions (without dots)
        """
        return [ext.lstrip('.') for ext in self.SUPPORTED_EXTENSIONS]
