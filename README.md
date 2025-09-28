# 🚀 RAG-Lib

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful and flexible Python library for building production-ready Retrieval-Augmented Generation (RAG) systems. RAG-Lib provides a complete toolkit for document processing, embedding, retrieval, and generation with enterprise-grade features.

## ✨ Key Features

### 🏗️ **Complete RAG Pipeline**
- **Document Processing**: PDF, DOCX, TXT, XLSX readers with structured extraction
- **Smart Chunking**: Length-based, recursive, and semantic chunking strategies
- **Vector Storage**: In-memory and Qdrant integration with more coming
- **Embedding Models**: OpenAI, SentenceTransformers with easy extensibility
- **Advanced Retrieval**: Semantic search with filtering and ranking
- **Generation**: Flexible generator interface with prompt templates

### 🔧 **Developer Experience**
- **Type Safety**: Full type annotations throughout
- **Configuration-Driven**: JSON/YAML configs for different environments
- **Factory Patterns**: Easy component creation and swapping
- **Rich Examples**: 15+ working examples covering all use cases
- **Error Handling**: Comprehensive exception hierarchy with clear messages

### 🎯 **Production Ready**
- **Modular Architecture**: Clean separation of concerns
- **Extensible Design**: Plugin-based architecture for custom components
- **Performance Optimized**: Batch processing and lazy loading
- **Robust Validation**: Input validation at every layer
- **Comprehensive Logging**: Structured logging with different levels

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# For development
pip install -e ".[dev]"
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Simple RAG Pipeline

```python
from rag_lib import (
    OpenAIEmbedding, InMemoryDocumentStore, SemanticRetriever, 
    RAGPipeline, NoOpRanker, Query, Document
)

# 1. Setup components
embedding_model = OpenAIEmbedding(model_name="text-embedding-3-small")
document_store = InMemoryDocumentStore(embedding_model=embedding_model)
retriever = SemanticRetriever("main", document_store, embedding_model)
ranker = NoOpRanker()

# 2. Create pipeline (generator implementation needed)
pipeline = RAGPipeline(retriever=retriever, ranker=ranker, generator=your_generator)

# 3. Add documents
documents = [
    Document(id="1", content="Python is a programming language"),
    Document(id="2", content="Machine learning uses algorithms to learn from data"),
    Document(id="3", content="RAG combines retrieval and generation for better AI responses")
]

document_store.create_index("knowledge_base", vector_size=1536)
document_store.add_documents("knowledge_base", documents)

# 4. Query the system
query = Query(content="How does machine learning work?")
response = pipeline.query(query, "knowledge_base", top_k=5)

print(f"Generated response: {response.generated_text}")
print(f"Retrieved {len(response.retrieved_documents)} relevant documents")
```

### Document Processing Pipeline

```python
from rag_lib.pipelines.data_pipeline import create_pipeline_from_config
from rag_lib.document_stores import InMemoryDocumentStore
from rag_lib.models.embedding.sbert_embedding import SBERTEmbedding

# 1. Setup
embedding_model = SBERTEmbedding()
document_store = InMemoryDocumentStore(embedding_model=embedding_model)

# 2. Configuration-driven pipeline
config = {
    "chunker": {
        "type": "recursive",
        "overlap_tokens": 50,
        "create_parent_chunks": True
    },
    "enable_progress": True,
    "batch_size": 10
}

pipeline = create_pipeline_from_config(config, document_store)

# 3. Process documents
file_paths = ["document1.pdf", "document2.docx", "data.xlsx"]
results = pipeline.process_files(
    file_paths=file_paths,
    index_name="my_documents",
    metadata={"source": "company_docs", "version": "2024"}
)

print(f"Processed {results.total_files} files")
print(f"Created {results.total_chunks} chunks")
print(f"Processing time: {results.total_time:.2f}s")
```

### Factory Pattern Usage

```python
from rag_lib.chunkers import create_chunker, get_available_chunker_types
from rag_lib.readers import create_multi_format_reader

# 1. Explore available options
chunker_types = get_available_chunker_types()
print(f"Available chunkers: {chunker_types}")

# 2. Create components with factory
chunker = create_chunker("semantic_sentence_transformer", {
    "model_name": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.7,
    "max_tokens": 512
})

reader = create_multi_format_reader()

# 3. Use with any file type
file_content = reader.read("any_document.pdf")
chunks = chunker.chunk_file_content(file_content)
print(f"Created {len(chunks)} chunks from {file_content.file_path}")
```

## 📚 Documentation

### Getting Started
- [📥 Installation Guide](INSTALL.md) - Complete setup instructions
- [🎯 Quick Tutorial](docs/tutorials/basic_rag_pipeline.md) - Build your first RAG system
- [⚙️ Configuration Guide](docs/configuration.md) - Configure for your use case

### API Reference
- [📖 Core Components](docs/api/core.md) - Documents, Queries, and Schemas
- [🔍 Retrievers](docs/api/retrievers.md) - Semantic and hybrid retrieval
- [📄 Document Stores](docs/api/document_stores.md) - Vector databases and storage
- [✂️ Chunkers](docs/api/chunkers.md) - Text chunking strategies
- [📚 Readers](docs/api/readers.md) - Document format support

### Tutorials & Guides
- [🏗️ Building Production RAG](docs/tutorials/production_rag.md)
- [🎨 Custom Components](docs/tutorials/custom_components.md)
- [📊 Performance Optimization](docs/tutorials/performance.md)
- [🐛 Debugging Guide](docs/tutorials/debugging.md)

### Examples
- [📁 Basic Examples](examples/) - Simple use cases
- [🔧 Advanced Examples](examples/advanced/) - Complex scenarios
- [🏭 Production Examples](examples/production/) - Real-world implementations

## 🏗️ Architecture

RAG-Lib follows a modular architecture with clear interfaces:

```
📦 RAG-Lib Architecture
├── 📄 Document Processing
│   ├── Readers (PDF, DOCX, TXT, XLSX)
│   ├── Chunkers (Length, Recursive, Semantic)
│   └── Data Pipelines
├── 🧠 Embedding & Storage  
│   ├── Embedding Models (OpenAI, SentenceTransformers)
│   ├── Document Stores (InMemory, Qdrant)
│   └── Vector Operations
├── 🔍 Retrieval & Ranking
│   ├── Retrievers (Semantic, Hybrid)
│   ├── Rankers (Cross-encoder, Custom)
│   └── Filtering & Search
├── 🎯 Generation
│   ├── Generators (LLM Integration)
│   ├── Prompt Templates
│   └── Response Processing
└── 🚀 Orchestration
    ├── RAG Pipelines
    ├── Configuration Management
    └── Factory Patterns
```

## 🎛️ Component Overview

### 📄 Document Readers
```python
from rag_lib.readers import PdfReader, DocxReader, get_reader_for_file

# Automatic format detection
reader = get_reader_for_file("document.pdf")
content = reader.read("document.pdf")

# Access structured elements
for element in content.elements:
    if element.type == ElementType.TEXT:
        print(f"Text: {element.content}")
    elif element.type == ElementType.TABLE:
        print(f"Table: {element.content}")
```

### ✂️ Smart Chunking
```python
from rag_lib.chunkers import create_chunker

# Semantic chunking with sentence similarity
chunker = create_chunker("semantic_sentence_transformer", {
    "model_name": "all-mpnet-base-v2",
    "similarity_threshold": 0.8,
    "max_tokens": 1024
})

chunks = chunker.chunk_text("Long document text here...")
```

### 🔍 Advanced Retrieval
```python
from rag_lib.retrievers import SemanticRetriever

retriever = SemanticRetriever("advanced", document_store, embedding_model)

# Filtered search with metadata
results = retriever.retrieve(
    query="machine learning", 
    index_name="docs",
    top_k=10,
    filters={"category": "AI", "year": 2024}
)
```

### 🎯 Generation with Templates
```python
from rag_lib.generators import PromptTemplate, DEFAULT_QA_TEMPLATE

template = PromptTemplate(
    template="Based on the following context: {context}\n\nAnswer: {query}",
    input_variables=["context", "query"]
)

prompt = template.format(
    context="Retrieved document content...",
    query="Your question here"
)
```

## 🔧 Configuration

RAG-Lib supports configuration-driven development:

```json
{
  "embedding_model": {
    "type": "openai",
    "model_name": "text-embedding-3-small"
  },
  "document_store": {
    "type": "qdrant",
    "url": "http://localhost:6333"
  },
  "chunker": {
    "type": "recursive",
    "max_tokens": 1024,
    "overlap_tokens": 100
  },
  "retriever": {
    "type": "semantic",
    "top_k": 10,
    "similarity_threshold": 0.7
  }
}
```

Load and use configurations:

```python
from rag_lib.pipelines import create_pipeline_from_config

pipeline = create_pipeline_from_config("config.json")
```

## 🌟 Advanced Features

### 🔄 Batch Processing
```python
# Process multiple queries efficiently
queries = ["Query 1", "Query 2", "Query 3"]
responses = pipeline.batch_query(queries, "index_name", top_k=5)
```

### 📊 Performance Monitoring
```python
# Built-in timing and metrics
response = pipeline.query(query, "index_name")
print(f"Retrieval time: {response.retrieval_time:.3f}s")
print(f"Generation time: {response.generation_time:.3f}s")
print(f"Total time: {response.total_time:.3f}s")
```

### 🎨 Custom Components
```python
from rag_lib.retrievers import BaseRetriever

class CustomRetriever(BaseRetriever):
    def retrieve(self, query, index_name, top_k=10, **kwargs):
        # Your custom retrieval logic
        return results
```

## 📋 Examples

### Basic RAG System
- [Simple Pipeline](examples/simple_pipeline_config_example.py)
- [Document Processing](examples/readers_example.py)
- [Embedding Models](examples/openai_llm_example.py)

### Advanced Use Cases  
- [Semantic Chunking](examples/semantic_chunker_example.py)
- [Cross-encoder Ranking](examples/cross_encoder_ranker_example.py)
- [Qdrant Integration](examples/qdrant_store_example.py)

### Production Ready
- [Configuration Management](examples/config_from_file_example.py)
- [Data Pipeline](examples/data_pipeline_config_example.py)
- [Component Testing](examples/test_rag_components.py)

## 🚀 Performance

RAG-Lib is optimized for both development and production:

- **Memory Efficient**: Lazy loading and streaming processing
- **Scalable**: Batch operations and async support (coming soon)
- **Fast**: Optimized vector operations and caching
- **Robust**: Comprehensive error handling and recovery

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/tungnk99/rag-lib.git
cd rag-lib
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v --cov=rag_lib
python examples/test_rag_components.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Inspiration for modular design
- [Sentence Transformers](https://www.sbert.net/) - Excellent embedding models
- [Qdrant](https://qdrant.tech/) - High-performance vector database

## 📞 Support

- **📚 Documentation**: [docs/](docs/)
- **💬 Discussions**: [GitHub Discussions](https://github.com/tungnk99/rag-lib/discussions)
- **🐛 Issues**: [GitHub Issues](https://github.com/tungnk99/rag-lib/issues)
- **📧 Email**: tungnk.hust@gmail.com

---

**Built with ❤️ for the RAG community**
