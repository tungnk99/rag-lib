# RAG-Lib

A lightweight Python library for building Retrieval-Augmented Generation (RAG) systems with flexible retrieval and LLM integration.

## Quick Start

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from rag_lib import OpenAIEmbedding, Query, Document

# Initialize embedding model
embedding_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    api_key="your-openai-api-key"  # or set OPENAI_API_KEY env var
)

# Encode a query
query = Query(content="What is machine learning?")
embedding_model.encode_query(query)

print(f"Query embedding dimension: {len(query.embedding)}")

# Encode documents
documents = [
    Document(id="doc1", content="Machine learning is a subset of AI"),
    Document(id="doc2", content="Python is a programming language")
]

embedding_model.batch_encode_documents(documents)
```

## Features

- 🚀 **Easy to use**: Simple API for embedding text into vectors
- 🔌 **Flexible**: Support for multiple embedding providers (OpenAI, more coming)
- 📊 **Efficient**: Batch processing for better performance
- 🛡️ **Robust**: Comprehensive error handling and validation
- 🧪 **Well-tested**: Extensive test coverage

## Installation Options

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Documentation

- [Installation Guide](INSTALL.md)
- [API Reference](docs/) (coming soon)
- [Examples](examples/) (coming soon)

## Requirements

- Python 3.8+
- OpenAI API key (for OpenAI embedding models)

## License

MIT License - see [LICENSE](LICENSE) file for details.
