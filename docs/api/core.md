# Core Components API Reference

This document covers the core components and data structures in RAG-Lib.

## Table of Contents
- [Schemas](#schemas)
- [Query](#query)
- [Document](#document)
- [RetrievalResult](#retrievalresult)
- [RAGResponse](#ragresponse)
- [Utility Functions](#utility-functions)

## Schemas

The core schemas define the fundamental data structures used throughout RAG-Lib.

### Module: `rag_lib.schemas.schema`

Import the core schemas:
```python
from rag_lib.schemas.schema import (
    Query, Document, RetrievalResult, RAGResponse,
    Element, FileContent, ElementType,
    create_query, create_document
)
```

## Query

Represents a user query in the RAG system.

### Class Definition

```python
@dataclass
class Query:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | `str` | The main query text content |
| `metadata` | `Dict[str, Any]` | Additional metadata (query_id, timestamp, filters, etc.) |
| `embedding` | `Optional[List[float]]` | Query embedding vector if computed |

### Methods

#### `__post_init__()`
Validates and sets default metadata after initialization.

**Automatically sets:**
- `query_id`: Unique UUID if not provided
- `timestamp`: Current ISO timestamp if not provided  
- `max_results`: Default value of 10 if not provided

#### `to_dict() -> Dict[str, Any]`
Convert query to dictionary representation.

**Returns:**
- `Dict[str, Any]`: Dictionary with content, metadata, and embedding

#### `from_dict(cls, data: Dict[str, Any]) -> 'Query'`
Create Query from dictionary representation.

**Parameters:**
- `data`: Dictionary containing query data

**Returns:**
- `Query`: Created query object

### Usage Examples

#### Basic Query Creation
```python
from rag_lib.schemas.schema import Query

# Simple query
query = Query(content="What is machine learning?")
print(f"Query ID: {query.metadata['query_id']}")
print(f"Timestamp: {query.metadata['timestamp']}")

# Query with metadata
query = Query(
    content="What is deep learning?",
    metadata={
        "user_id": "user123",
        "session_id": "session456",
        "max_results": 5,
        "filters": {"category": "ai"}
    }
)
```

#### Working with Embeddings
```python
# Query with pre-computed embedding
query = Query(
    content="Python programming",
    embedding=[0.1, 0.2, 0.3, ...]  # Your embedding vector
)

# Check if embedding exists
if query.embedding:
    print(f"Embedding dimension: {len(query.embedding)}")
else:
    print("No embedding computed yet")
```

#### Serialization
```python
# Convert to dictionary
query_dict = query.to_dict()

# Recreate from dictionary
restored_query = Query.from_dict(query_dict)
```

### Validation

Queries are automatically validated:

```python
# This will raise ValueError
try:
    Query(content="")  # Empty content
except ValueError as e:
    print(f"Validation error: {e}")

try:
    Query(content="   ")  # Only whitespace
except ValueError as e:
    print(f"Validation error: {e}")
```

## Document

Represents a document in the RAG system.

### Class Definition

```python
@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier for the document |
| `content` | `str` | The main document content/text |
| `metadata` | `Dict[str, Any]` | Additional metadata about the document |
| `embedding` | `Optional[List[float]]` | Document embedding vector if computed |

### Methods

#### `__post_init__()`
Validates document and sets default metadata.

**Automatically sets:**
- `created_at`: Current ISO timestamp if not provided
- `updated_at`: Current ISO timestamp if not provided
- `doc_type`: Default value "text" if not provided

#### `to_dict() -> Dict[str, Any]`
Convert document to dictionary representation.

#### `from_dict(cls, data: Dict[str, Any]) -> 'Document'`
Create Document from dictionary representation.

### Usage Examples

#### Basic Document Creation
```python
from rag_lib.schemas.schema import Document

# Simple document
doc = Document(
    id="doc_001",
    content="Python is a versatile programming language."
)

# Document with rich metadata
doc = Document(
    id="research_paper_123",
    content="Abstract: This paper presents a novel approach to...",
    metadata={
        "title": "Novel AI Approach",
        "authors": ["John Doe", "Jane Smith"],
        "category": "research",
        "year": 2024,
        "doi": "10.1000/xyz123",
        "page_count": 12,
        "source": "academic_journal"
    }
)
```

#### Working with Chunks
```python
# Create document chunks
parent_doc = Document(
    id="book_chapter_5",
    content="Full chapter content here...",
    metadata={"source": "book.pdf", "chapter": 5}
)

# Create chunks with parent reference
chunks = []
for i, chunk_text in enumerate(chunked_texts):
    chunk = Document(
        id=f"{parent_doc.id}_chunk_{i}",
        content=chunk_text,
        metadata={
            **parent_doc.metadata,
            "parent_id": parent_doc.id,
            "chunk_index": i,
            "chunk_type": "paragraph"
        }
    )
    chunks.append(chunk)
```

#### Document Updates
```python
# Update document metadata
doc.metadata["last_accessed"] = datetime.now().isoformat()
doc.metadata["access_count"] = doc.metadata.get("access_count", 0) + 1

# Add embedding after computation
doc.embedding = embedding_model.encode(doc.content)
```

## RetrievalResult

Represents a retrieval result containing a document and its relevance score.

### Class Definition

```python
@dataclass
class RetrievalResult:
    document: Document
    score: float
    rank: int = 0
    explanation: Optional[str] = None
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `document` | `Document` | The retrieved document |
| `score` | `float` | Relevance score (0-1, higher is more relevant) |
| `rank` | `int` | Rank in the retrieval results (1-based) |
| `explanation` | `Optional[str]` | Explanation of why this document was retrieved |

### Usage Examples

#### Creating Retrieval Results
```python
from rag_lib.schemas.schema import RetrievalResult, Document

doc = Document(id="doc1", content="Machine learning content...")

result = RetrievalResult(
    document=doc,
    score=0.85,
    rank=1,
    explanation="High semantic similarity with query terms"
)
```

#### Working with Results
```python
# Sort results by score
results = [result1, result2, result3]
sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

# Update ranks
for i, result in enumerate(sorted_results, 1):
    result.rank = i

# Filter by score threshold
high_quality_results = [r for r in results if r.score >= 0.7]
```

## RAGResponse

Represents the complete response from a RAG system.

### Class Definition

```python
@dataclass
class RAGResponse:
    query: Query
    generated_text: str
    retrieved_documents: List[RetrievalResult] = field(default_factory=list)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    total_time: Optional[float] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
```

### Methods

#### `get_top_documents(n: int = 3) -> List[RetrievalResult]`
Get top N retrieved documents by score.

**Parameters:**
- `n`: Number of top documents to return

**Returns:**
- `List[RetrievalResult]`: Top documents sorted by relevance score

### Usage Examples

#### Creating RAG Responses
```python
from rag_lib.schemas.schema import RAGResponse, Query

response = RAGResponse(
    query=Query(content="What is AI?"),
    generated_text="Artificial Intelligence (AI) is...",
    retrieved_documents=[result1, result2, result3],
    generation_metadata={
        "model": "gpt-3.5-turbo",
        "tokens_used": 150,
        "temperature": 0.1
    },
    total_time=2.5,
    retrieval_time=1.2,
    generation_time=1.3
)
```

#### Analyzing Responses
```python
# Get performance metrics
print(f"Total processing time: {response.total_time:.2f}s")
print(f"Retrieval efficiency: {response.retrieval_time/response.total_time:.1%}")
print(f"Generation efficiency: {response.generation_time/response.total_time:.1%}")

# Get top documents used
top_docs = response.get_top_documents(n=2)
for doc in top_docs:
    print(f"- {doc.document.id}: {doc.score:.3f}")

# Check response quality
if len(response.retrieved_documents) == 0:
    print("Warning: No documents retrieved")
elif all(r.score < 0.5 for r in response.retrieved_documents):
    print("Warning: Low relevance scores")
```

## Utility Functions

Convenience functions for creating and working with core objects.

### `create_query(content: str, metadata: Optional[Dict] = None, embedding: Optional[List[float]] = None) -> Query`

Create a Query object with convenience.

```python
from rag_lib.schemas.schema import create_query

query = create_query(
    content="What is machine learning?",
    metadata={"user": "john", "session": "abc123"}
)
```

### `create_document(doc_id: str, content: str, metadata: Optional[Dict] = None, embedding: Optional[List[float]] = None) -> Document`

Create a Document object with convenience.

```python
from rag_lib.schemas.schema import create_document

doc = create_document(
    doc_id="doc1",
    content="Machine learning is...",
    metadata={"source": "textbook", "chapter": 1}
)
```

### `validate_embedding(embedding: List[float], expected_dim: Optional[int] = None) -> bool`

Validate embedding vector format and dimensions.

```python
from rag_lib.schemas.schema import validate_embedding

# Validate embedding
is_valid = validate_embedding([0.1, 0.2, 0.3], expected_dim=3)
if not is_valid:
    print("Invalid embedding format")

# Check dimension mismatch
is_valid = validate_embedding([0.1, 0.2], expected_dim=3)  # False
```

### `batch_create_documents(documents_data: List[Dict[str, Any]]) -> List[Document]`

Create multiple documents from dictionaries.

```python
from rag_lib.schemas.schema import batch_create_documents

docs_data = [
    {"id": "doc1", "content": "Content 1", "metadata": {"type": "article"}},
    {"id": "doc2", "content": "Content 2", "metadata": {"type": "paper"}},
]

documents = batch_create_documents(docs_data)
```

## Element and FileContent

For document processing and structured content extraction.

### ElementType Enum

```python
from rag_lib.schemas.schema import ElementType

# Available types
ElementType.TEXT    # Regular text content
ElementType.IMAGE   # Images, charts, figures  
ElementType.TABLE   # Tables, structured data
```

### Element Class

Represents an element extracted from a document.

```python
from rag_lib.schemas.schema import Element, ElementType

element = Element(
    type=ElementType.TEXT,
    content="This is a paragraph from the document.",
    metadata={"page_number": 1, "confidence": 0.95}
)
```

### FileContent Class

Represents content extracted from a file.

```python
from rag_lib.schemas.schema import FileContent, Element, ElementType

# Create file content
file_content = FileContent(
    file_path="document.pdf",
    elements=[
        Element(ElementType.TEXT, "First paragraph", {"page": 1}),
        Element(ElementType.TABLE, "| Col1 | Col2 |", {"page": 2}),
    ],
    metadata={"file_size": 1024, "processed_at": "2024-01-01"}
)

# Access content
text_content = file_content.get_text_content()
total_elements = file_content.get_total_elements_count()
elements_by_type = file_content.get_elements_count_by_type()
```

## Error Handling

All core classes include validation and will raise appropriate errors:

```python
try:
    # Invalid document
    doc = Document(id="", content="test")  # Empty ID
except ValueError as e:
    print(f"Document validation error: {e}")

try:
    # Invalid retrieval result
    result = RetrievalResult(doc, score=1.5)  # Score > 1
except ValueError as e:
    print(f"Result validation error: {e}")
```

## Best Practices

### Metadata Usage
```python
# Good metadata structure
metadata = {
    # Required for tracking
    "source": "company_docs",
    "version": "v2.1",
    
    # Useful for filtering
    "category": "technical",
    "department": "engineering",
    "access_level": "internal",
    
    # Helpful for ranking
    "quality_score": 0.9,
    "freshness": "2024-01-15",
    "popularity": 150,
    
    # Debugging information
    "processing_time": 2.3,
    "chunk_method": "semantic"
}
```

### Performance Considerations
```python
# Efficient document creation
documents = []
for i, text in enumerate(large_text_list):
    doc = Document(
        id=f"doc_{i}",
        content=text,
        metadata={"batch_id": batch_id}  # Minimal metadata
    )
    documents.append(doc)

# Batch operations
docs = batch_create_documents(doc_data_list)
```

This covers the core components API. For specific component APIs (retrievers, chunkers, etc.), see the respective API documentation files.
