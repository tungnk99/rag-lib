# Document Readers

This module provides document readers for various file formats in the RAG system. Each reader extracts content from specific file types and returns structured elements (text, images, tables).

## Available Readers

### BaseReader
Abstract base class that defines the interface for all document readers.

### TxtReader
Reads plain text files (`.txt`, `.text`).

**Features:**
- Extract text content
- Optional splitting by lines
- Configurable encoding
- Line counting and word counting metadata

**Example:**
```python
from rag_lib.readers import TxtReader

reader = TxtReader(split_by_lines=True, encoding='utf-8')
elements = reader.read_with_metadata('document.txt')
```

### PdfReader
Reads PDF documents (`.pdf`).

**Features:**
- Extract text content using PyPDF2 or pdfplumber
- Extract tables (requires pdfplumber)
- Extract images (requires pdf2image)
- Split by pages or combine all content
- Metadata including page numbers

**Dependencies:**
- `PyPDF2` or `pdfplumber` for text extraction
- `pdfplumber` for table extraction
- `pdf2image` for image extraction

**Example:**
```python
from rag_lib.readers import PdfReader

reader = PdfReader(
    extract_tables=True,
    extract_images=False,
    split_by_pages=True
)
elements = reader.read_with_metadata('document.pdf')
```

### DocxReader
Reads Microsoft Word documents (`.docx`).

**Features:**
- Extract text content from paragraphs
- Extract tables
- Extract images
- Document metadata (author, title, etc.)
- Style information

**Dependencies:**
- `python-docx` for document parsing
- `Pillow` for image handling (optional)

**Example:**
```python
from rag_lib.readers import DocxReader

reader = DocxReader(
    extract_tables=True,
    extract_images=False,
    split_by_paragraphs=True
)
elements = reader.read_with_metadata('document.docx')
```

### XlsxReader
Reads Excel spreadsheet files (`.xlsx`, `.xlsm`).

**Features:**
- Extract data from multiple worksheets
- Handle formulas and various data types
- Extract comments and notes
- Support for specific range reading
- Configurable row/column limits
- Workbook metadata extraction

**Dependencies:**
- `openpyxl` for Excel file parsing

**Example:**
```python
from rag_lib.readers import XlsxReader

reader = XlsxReader(
    include_formulas=False,
    include_empty_cells=False,
    max_rows=1000,
    sheet_names=['Sheet1', 'Data']  # Specific sheets
)
elements = reader.read_with_metadata('spreadsheet.xlsx')

# Read specific range
range_elements = reader.read_specific_range(
    'spreadsheet.xlsx', 'Sheet1', 
    start_row=1, end_row=10,
    start_col=1, end_col=5
)
```

### MultiFormatReader
Handles multiple file formats by automatically selecting appropriate readers.

**Example:**
```python
from rag_lib.readers import create_multi_format_reader

reader = create_multi_format_reader()
elements = reader.read('document.pdf')  # Automatically uses PdfReader
```

## Element Types

All readers return a list of `Element` objects with the following types:

- **TEXT**: Text content from paragraphs, pages, or lines
- **IMAGE**: Images extracted from documents (as base64)
- **TABLE**: Tables extracted from documents (as text representation)

## Element Structure

Each `Element` contains:
- `type`: ElementType (TEXT, IMAGE, TABLE)
- `content`: The actual content (string)
- `metadata`: Additional information (page numbers, coordinates, etc.)

## Installation

Install required dependencies:

```bash
# For PDF reading
pip install PyPDF2
# or
pip install pdfplumber

# For DOCX reading  
pip install python-docx

# For XLSX reading
pip install openpyxl

# For image extraction from PDFs
pip install pdf2image Pillow
```

## Usage Examples

### Basic Usage
```python
from rag_lib.readers import get_reader_for_file

# Automatic reader selection
reader = get_reader_for_file('document.pdf')
elements = reader.read_with_metadata('document.pdf')

for element in elements:
    print(f"Type: {element.type.value}")
    print(f"Content: {element.content[:100]}...")
    print(f"Metadata: {element.metadata}")
```

### Processing Multiple Files
```python
from rag_lib.readers import create_multi_format_reader

reader = create_multi_format_reader()
files = ['doc1.txt', 'doc2.pdf', 'doc3.docx', 'data.xlsx']

all_elements = []
for file_path in files:
    try:
        elements = reader.read(file_path)
        all_elements.extend(elements)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
```

### Custom Configuration
```python
from rag_lib.readers import PdfReader, DocxReader, TxtReader, XlsxReader

# PDF with table extraction
pdf_reader = PdfReader(
    extract_tables=True,
    extract_images=True,
    split_by_pages=False
)

# DOCX with paragraph splitting
docx_reader = DocxReader(
    split_by_paragraphs=True,
    extract_tables=True
)

# XLSX with specific configuration
xlsx_reader = XlsxReader(
    include_formulas=True,
    max_rows=500,
    sheet_names=['Data', 'Summary']
)

# Text with line splitting
txt_reader = TxtReader(
    split_by_lines=True,
    encoding='utf-8'
)
```

## Error Handling

Readers provide detailed error messages for common issues:
- File not found
- Unsupported file format
- Missing dependencies
- Encoding issues
- Corrupted files

## Extending Readers

To create a custom reader:

```python
from rag_lib.readers import BaseReader
from rag_lib.schemas import Element, ElementType, create_element

class CustomReader(BaseReader):
    SUPPORTED_EXTENSIONS = {'.custom'}
    
    def supports_format(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def read(self, file_path: str) -> List[Element]:
        # Implement your reading logic
        content = "extracted content"
        element = create_element(ElementType.TEXT, content)
        return [element]
```
