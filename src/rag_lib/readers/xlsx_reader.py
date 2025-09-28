"""
XLSX file reader for reading Excel spreadsheet files.

This module provides functionality to read XLSX files and extract
their content as table and text elements using openpyxl.

Dependencies:
    - openpyxl for Excel file parsing
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import time
from ._base import BaseReader
from ..schemas.schema import Element, ElementType, create_element, FileContent


class XlsxReader(BaseReader):
    """
    Reader for Excel spreadsheet files (.xlsx, .xlsm).
    
    This reader extracts data from Excel worksheets and returns them as
    table elements. It can process multiple sheets and handle various
    data types including formulas.
    """
    
    SUPPORTED_EXTENSIONS = {'.xlsx', '.xlsm'}
    
    def __init__(self, 
                 include_formulas: bool = False,
                 include_empty_cells: bool = False,
                 max_rows: Optional[int] = None,
                 max_cols: Optional[int] = None,
                 sheet_names: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize the XLSX reader.
        
        Args:
            include_formulas (bool): Whether to include formula text instead of values
            include_empty_cells (bool): Whether to include empty cells in output
            max_rows (Optional[int]): Maximum number of rows to read per sheet
            max_cols (Optional[int]): Maximum number of columns to read per sheet
            sheet_names (Optional[List[str]]): Specific sheet names to read (None = all sheets)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.include_formulas = include_formulas
        self.include_empty_cells = include_empty_cells
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.sheet_names = sheet_names
        
        # Check for required dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import openpyxl
            self.openpyxl_available = True
        except ImportError:
            raise ImportError(
                "XLSX reading requires openpyxl. "
                "Install with: pip install openpyxl"
            )
    
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
        Read an XLSX file and extract its content.
        
        Args:
            file_path (str): Path to the XLSX file to read
            
        Returns:
            FileContent: File content object containing extracted elements and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other reading errors
        """
        start_time = time.time()
        
        try:
            import openpyxl
            from openpyxl.utils import get_column_letter
            
            elements = []
            
            # Load workbook
            workbook = openpyxl.load_workbook(
                file_path, 
                data_only=not self.include_formulas,
                read_only=True
            )
            
            # Determine which sheets to process
            sheets_to_process = self.sheet_names if self.sheet_names else workbook.sheetnames
            
            for sheet_name in sheets_to_process:
                if sheet_name not in workbook.sheetnames:
                    print(f"Warning: Sheet '{sheet_name}' not found in workbook")
                    continue
                
                sheet = workbook[sheet_name]
                sheet_elements = self._extract_sheet_data(sheet, sheet_name)
                elements.extend(sheet_elements)
            
            workbook.close()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get workbook properties for metadata
            xlsx_metadata = self.get_workbook_info(file_path)
            xlsx_metadata.update({
                'include_formulas': self.include_formulas,
                'include_empty_cells': self.include_empty_cells,
                'max_rows': self.max_rows,
                'max_cols': self.max_cols,
                'processed_sheets': sheets_to_process,
                'total_elements': len(elements)
            })
            
            # Create and return FileContent
            return self.create_file_content(
                file_path=file_path,
                elements=elements,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Error reading XLSX file {file_path}: {str(e)}")
    
    def _extract_sheet_data(self, sheet, sheet_name: str) -> List[Element]:
        """Extract data from a single worksheet."""
        elements = []
        
        # Get sheet dimensions
        max_row = sheet.max_row
        max_col = sheet.max_column
        
        # Apply limits if specified
        if self.max_rows:
            max_row = min(max_row, self.max_rows)
        if self.max_cols:
            max_col = min(max_col, self.max_cols)
        
        if max_row == 0 or max_col == 0:
            return elements
        
        # Extract table data
        table_data = []
        for row_idx in range(1, max_row + 1):
            row_data = []
            for col_idx in range(1, max_col + 1):
                cell = sheet.cell(row=row_idx, column=col_idx)
                cell_value = self._get_cell_value(cell)
                
                # Handle empty cells
                if cell_value is None or cell_value == "":
                    if self.include_empty_cells:
                        row_data.append("")
                    else:
                        row_data.append("")
                else:
                    row_data.append(str(cell_value))
            
            # Only add non-empty rows (unless including empty cells)
            if self.include_empty_cells or any(cell.strip() for cell in row_data):
                table_data.append(row_data)
        
        if table_data:
            # Create table element
            table_text = self._table_to_text(table_data)
            
            metadata = {
                'sheet_name': sheet_name,
                'rows': len(table_data),
                'columns': len(table_data[0]) if table_data else 0,
                'max_row': max_row,
                'max_col': max_col,
                'extraction_method': 'openpyxl'
            }
            
            element = create_element(
                element_type=ElementType.TABLE,
                content=table_text,
                metadata=metadata
            )
            elements.append(element)
        
        # Extract text from merged cells and comments if any
        text_elements = self._extract_text_elements(sheet, sheet_name)
        elements.extend(text_elements)
        
        return elements
    
    def _get_cell_value(self, cell) -> Any:
        """Get the appropriate value from a cell."""
        if self.include_formulas and hasattr(cell, 'formula') and cell.formula:
            return cell.formula
        else:
            return cell.value
    
    def _extract_text_elements(self, sheet, sheet_name: str) -> List[Element]:
        """Extract text elements like comments and notes."""
        elements = []
        
        try:
            # Extract comments
            comments = []
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.comment:
                        comment_text = cell.comment.text
                        if comment_text and comment_text.strip():
                            comments.append(f"Cell {cell.coordinate}: {comment_text}")
            
            if comments:
                comments_text = "\n".join(comments)
                metadata = {
                    'sheet_name': sheet_name,
                    'element_type': 'comments',
                    'comment_count': len(comments),
                    'extraction_method': 'openpyxl'
                }
                
                element = create_element(
                    element_type=ElementType.TEXT,
                    content=comments_text,
                    metadata=metadata
                )
                elements.append(element)
        
        except Exception as e:
            print(f"Warning: Error extracting text elements from sheet {sheet_name}: {e}")
        
        return elements
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to text representation."""
        if not table_data:
            return ""
        
        # Use tab-separated format for better Excel compatibility
        lines = []
        for row in table_data:
            lines.append("\t".join(row))
        
        return "\n".join(lines)
    
    def get_workbook_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get workbook properties and metadata.
        
        Args:
            file_path (str): Path to the XLSX file
            
        Returns:
            Dict[str, Any]: Workbook properties
        """
        try:
            import openpyxl
            
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            
            # Basic workbook info
            info = {
                'sheet_names': workbook.sheetnames,
                'sheet_count': len(workbook.sheetnames),
                'active_sheet': workbook.active.title if workbook.active else None
            }
            
            # Get properties if available
            try:
                props = workbook.properties
                info.update({
                    'title': props.title,
                    'creator': props.creator,
                    'description': props.description,
                    'subject': props.subject,
                    'created': props.created,
                    'modified': props.modified,
                    'category': props.category,
                    'keywords': props.keywords
                })
            except Exception:
                pass
            
            # Get sheet details
            sheet_details = {}
            for sheet_name in workbook.sheetnames:
                try:
                    sheet = workbook[sheet_name]
                    sheet_details[sheet_name] = {
                        'max_row': sheet.max_row,
                        'max_column': sheet.max_column,
                        'title': sheet.title
                    }
                except Exception:
                    pass
            
            info['sheet_details'] = sheet_details
            
            workbook.close()
            return info
            
        except Exception as e:
            print(f"Warning: Could not extract workbook properties: {str(e)}")
            return {}
    
    def read_specific_range(self, file_path: str, sheet_name: str, 
                          start_row: int, end_row: int, 
                          start_col: int, end_col: int) -> FileContent:
        """
        Read a specific range from a worksheet.
        
        Args:
            file_path (str): Path to the XLSX file
            sheet_name (str): Name of the worksheet
            start_row (int): Starting row (1-based)
            end_row (int): Ending row (1-based)
            start_col (int): Starting column (1-based)
            end_col (int): Ending column (1-based)
            
        Returns:
            FileContent: File content object containing extracted elements from the specified range
        """
        try:
            import openpyxl
            
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            
            if sheet_name not in workbook.sheetnames:
                raise ValueError(f"Sheet '{sheet_name}' not found in workbook")
            
            sheet = workbook[sheet_name]
            
            # Extract data from specified range
            table_data = []
            for row_idx in range(start_row, end_row + 1):
                row_data = []
                for col_idx in range(start_col, end_col + 1):
                    try:
                        cell = sheet.cell(row=row_idx, column=col_idx)
                        cell_value = self._get_cell_value(cell)
                        row_data.append(str(cell_value) if cell_value is not None else "")
                    except Exception:
                        row_data.append("")
                table_data.append(row_data)
            
            workbook.close()
            
            elements = []
            if table_data:
                table_text = self._table_to_text(table_data)
                
                metadata = {
                    'sheet_name': sheet_name,
                    'range': f"{start_row}:{end_row},{start_col}:{end_col}",
                    'rows': len(table_data),
                    'columns': len(table_data[0]) if table_data else 0,
                    'extraction_method': 'openpyxl'
                }
                
                element = create_element(
                    element_type=ElementType.TABLE,
                    content=table_text,
                    metadata=metadata
                )
                elements.append(element)
            
            # Create FileContent
            range_metadata = {
                'sheet_name': sheet_name,
                'range': f"{start_row}:{end_row},{start_col}:{end_col}",
                'extraction_method': 'range_specific',
                'total_elements': len(elements)
            }
            
            return self.create_file_content(
                file_path=file_path,
                elements=elements,
                processing_time=None
            )
            
        except Exception as e:
            raise Exception(f"Error reading range from XLSX file: {str(e)}")
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported extensions (without dots)
        """
        return [ext.lstrip('.') for ext in self.SUPPORTED_EXTENSIONS]
