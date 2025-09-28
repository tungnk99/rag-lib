# Contributing to RAG-Lib

Thank you for your interest in contributing to RAG-Lib! We welcome contributions from the community and are grateful for any help you can provide.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get started
- **Be collaborative**: Work together and help each other
- **Be constructive**: Provide helpful feedback and suggestions

## Getting Started

### Types of Contributions

We welcome many types of contributions:

- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve docs, tutorials, and examples
- 🧪 **Testing**: Add or improve test coverage
- 🔧 **Code**: Fix bugs or implement new features
- 🎨 **Examples**: Create examples and tutorials
- 🌐 **Integrations**: Add new embedding models, document stores, etc.

### Before You Start

1. **Check existing issues**: Look for existing issues or discussions
2. **Create an issue**: For new features or major changes, create an issue first
3. **Start small**: Begin with documentation or small bug fixes
4. **Ask questions**: Don't hesitate to ask for help or clarification

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/rag-lib.git
cd rag-lib

# Add upstream remote
git remote add upstream https://github.com/tungnk99/rag-lib.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv-dev
source venv-dev/bin/activate  # On Windows: venv-dev\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Run the component test
python examples/test_rag_components.py

# Check code style
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/
```

## Contributing Guidelines

### Branch Naming

Use descriptive branch names:

```bash
# Feature branches
git checkout -b feature/add-pinecone-integration
git checkout -b feature/improve-chunking-performance

# Bug fix branches  
git checkout -b fix/memory-leak-in-retriever
git checkout -b fix/pdf-reader-encoding-issue

# Documentation branches
git checkout -b docs/api-reference-update
git checkout -b docs/tutorial-improvements
```

### Commit Messages

Follow conventional commit format:

```bash
# Format: type(scope): description
feat(chunkers): add semantic chunking with BERT embeddings
fix(readers): handle PDF files with special characters
docs(api): add examples for document store usage
test(retrievers): add unit tests for semantic retriever
refactor(core): improve error handling in base classes
```

Types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes (formatting, etc.)
- `chore`: Maintenance tasks

### Code Organization

Follow the existing project structure:

```
src/rag_lib/
├── __init__.py          # Main public API
├── schemas/             # Data models and schemas
├── models/              # Embedding and LLM models
│   ├── embedding/
│   └── llm/
├── readers/             # Document readers
├── chunkers/            # Text chunking strategies
├── document_stores/     # Vector databases and storage
├── retrievers/          # Retrieval components
├── rankers/             # Document ranking
├── generators/          # Text generation
├── pipelines/           # End-to-end pipelines
└── utils/               # Utility functions
```

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Keep your fork updated
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code, test, document ...

# Run tests and checks
pytest tests/ -v
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### 2. Commit and Push

```bash
# Stage and commit changes
git add .
git commit -m "feat(component): add new feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to GitHub and create a PR from your fork
2. Fill out the PR template completely
3. Link related issues
4. Request review from maintainers

### 4. PR Requirements

Your PR should include:

- [ ] **Clear description** of changes and motivation
- [ ] **Tests** for new functionality or bug fixes
- [ ] **Documentation** updates if needed
- [ ] **Examples** if adding new features
- [ ] **Type hints** for all new code
- [ ] **Error handling** with appropriate exceptions
- [ ] **Backward compatibility** (or clear breaking change notes)

### 5. Review Process

1. **Automated checks**: CI/CD will run tests and checks
2. **Code review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good example
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

class DocumentReader(ABC):
    """
    Abstract base class for document readers.
    
    This class defines the interface for reading various document formats
    and extracting structured content.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the document reader.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
    
    @abstractmethod
    def read(self, file_path: str) -> FileContent:
        """
        Read and extract content from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            FileContent: Extracted content with metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        pass
```

### Type Hints

All new code must include type hints:

```python
from typing import List, Dict, Optional, Union, Any

def process_documents(
    documents: List[Document],
    config: Optional[Dict[str, Any]] = None
) -> List[ProcessedDocument]:
    """Process a list of documents with optional configuration."""
    pass
```

### Error Handling

Use custom exceptions and proper error handling:

```python
class ChunkerError(Exception):
    """Base exception for chunker operations."""
    pass

class InvalidChunkSizeError(ChunkerError):
    """Raised when chunk size is invalid."""
    pass

def chunk_text(self, text: str, max_tokens: int) -> List[str]:
    """Chunk text into smaller pieces."""
    if max_tokens <= 0:
        raise InvalidChunkSizeError(f"max_tokens must be positive, got {max_tokens}")
    
    try:
        # Chunking logic here
        return chunks
    except Exception as e:
        raise ChunkerError(f"Failed to chunk text: {e}") from e
```

### Documentation

Every public function, class, and module should have docstrings:

```python
def semantic_search(
    self,
    query: Query,
    index_name: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[RetrievalResult]:
    """
    Perform semantic search for relevant documents.
    
    This method encodes the query using the embedding model and searches
    for the most similar documents in the specified index.
    
    Args:
        query: The search query
        index_name: Name of the index to search in
        top_k: Maximum number of results to return
        filters: Optional metadata filters to apply
        
    Returns:
        List of retrieval results sorted by relevance score
        
    Raises:
        IndexNotFoundError: If the specified index doesn't exist
        QueryEncodingError: If query encoding fails
        RetrievalError: If search operation fails
        
    Example:
        >>> retriever = SemanticRetriever(store, embedding_model)
        >>> query = Query("machine learning")
        >>> results = retriever.semantic_search(query, "docs", top_k=5)
        >>> print(f"Found {len(results)} relevant documents")
    """
```

## Testing

### Test Organization

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_chunkers.py
│   ├── test_readers.py
│   └── test_retrievers.py
├── integration/             # Integration tests
│   ├── test_pipelines.py
│   └── test_end_to_end.py
├── fixtures/               # Test data and fixtures
│   ├── sample_documents/
│   └── test_configs/
└── conftest.py            # Pytest configuration
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from rag_lib.chunkers import BaseLengthChunker
from rag_lib.schemas.schema import Document

class TestBaseLengthChunker:
    """Test suite for BaseLengthChunker."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking functionality."""
        chunker = BaseLengthChunker(max_tokens=10, overlap_tokens=2)
        text = "This is a sample text for testing chunking functionality."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk.split()) <= 10 for chunk in chunks)
    
    def test_chunk_text_empty_input(self):
        """Test chunking with empty input."""
        chunker = BaseLengthChunker()
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            chunker.chunk_text("")
    
    def test_chunk_text_invalid_max_tokens(self):
        """Test chunker initialization with invalid parameters."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            BaseLengthChunker(max_tokens=0)
    
    @patch('rag_lib.chunkers.base_length_chunker.tokenize')
    def test_chunk_text_with_mock(self, mock_tokenize):
        """Test chunking with mocked tokenizer."""
        mock_tokenize.return_value = ['token1', 'token2', 'token3']
        chunker = BaseLengthChunker(max_tokens=2)
        
        chunks = chunker.chunk_text("test text")
        
        mock_tokenize.assert_called_once_with("test text")
        assert len(chunks) >= 1
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_chunkers.py -v

# Run with coverage
pytest tests/ --cov=rag_lib --cov-report=html

# Run specific test
pytest tests/unit/test_chunkers.py::TestBaseLengthChunker::test_chunk_text_basic -v

# Run tests matching pattern
pytest tests/ -k "chunker" -v
```

## Documentation

### Types of Documentation

1. **API Documentation**: Docstrings in code
2. **Tutorials**: Step-by-step guides
3. **Examples**: Working code examples
4. **Reference**: Complete API reference

### Writing Documentation

```markdown
# Component Name

Brief description of what this component does.

## Overview

Detailed explanation of the component's purpose, when to use it,
and how it fits into the broader system.

## Basic Usage

```python
from rag_lib.component import ComponentName

# Simple example
component = ComponentName(config={"param": "value"})
result = component.process(input_data)
```

## Advanced Usage

More complex examples showing different configurations and use cases.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| param1    | str  | "default" | Description of param1 |
| param2    | int  | 100     | Description of param2 |

## API Reference

Detailed API documentation with all methods and parameters.
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation (when implemented)
cd docs/
sphinx-build -b html source build
```

## Adding New Components

### 1. Embedding Models

To add a new embedding model:

```python
# src/rag_lib/models/embedding/your_model.py
from ._base import EmbeddingModel

class YourEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, dimension=768)  # Set correct dimension
    
    def encode_text(self, text: str, **kwargs) -> List[float]:
        # Implement encoding logic
        return embedding
    
    def batch_encode_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        # Implement batch encoding
        return embeddings
```

### 2. Document Stores

To add a new document store:

```python
# src/rag_lib/document_stores/your_store.py
from ._base import BaseDocumentStore

class YourDocumentStore(BaseDocumentStore):
    def create_index(self, index_name: str, config: Optional[Dict] = None) -> bool:
        # Implement index creation
        pass
    
    def add_documents(self, index_name: str, documents: List[Document]) -> bool:
        # Implement document addition
        pass
    
    # Implement other required methods...
```

### 3. Update __init__.py Files

Add your new component to the appropriate `__init__.py`:

```python
# src/rag_lib/models/embedding/__init__.py
from .your_model import YourEmbeddingModel

__all__ = [..., "YourEmbeddingModel"]
```

### 4. Add Tests

Create comprehensive tests for your new component:

```python
# tests/unit/test_your_component.py
class TestYourComponent:
    def test_basic_functionality(self):
        # Test basic usage
        pass
    
    def test_error_handling(self):
        # Test error conditions
        pass
    
    def test_edge_cases(self):
        # Test edge cases
        pass
```

### 5. Add Examples

Create usage examples:

```python
# examples/your_component_example.py
"""
Example usage of YourComponent.
"""

def main():
    # Demonstrate usage
    pass

if __name__ == "__main__":
    main()
```

### 6. Update Documentation

- Add docstrings to your code
- Update relevant documentation
- Add to README if it's a major feature

## Community

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: tungnk.hust@gmail.com for direct contact

### Getting Help

1. **Check documentation**: Look at docs and examples first
2. **Search issues**: Check if your question has been asked
3. **Ask questions**: Create a GitHub discussion or issue
4. **Join development**: We welcome all levels of contributors

### Becoming a Maintainer

Regular contributors who demonstrate:
- High-quality code and documentation
- Helpful community engagement
- Understanding of the project goals
- Reliability in reviews and maintenance

May be invited to become maintainers.

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. Update version in `src/rag_lib/__version__.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release tag
5. Build and publish package
6. Update documentation

## Questions?

If you have questions about contributing:

1. Check this guide thoroughly
2. Look at existing code for examples
3. Create a GitHub discussion
4. Reach out to maintainers

Thank you for contributing to RAG-Lib! 🚀
