# RAG-Lib Documentation

Welcome to the RAG-Lib documentation! This comprehensive guide will help you get started with building powerful RAG (Retrieval-Augmented Generation) systems.

## 📚 Documentation Overview

### 🚀 Getting Started
- **[Installation Guide](../INSTALL.md)** - Complete setup instructions with troubleshooting
- **[Getting Started](getting_started.md)** - Build your first RAG system in minutes
- **[Configuration Guide](configuration.md)** - Configure RAG-Lib for your environment

### 📖 Tutorials & Guides
- **[Basic RAG Pipeline](tutorials/basic_rag_pipeline.md)** - Complete tutorial for building production-ready RAG systems
- **[Component Deep Dive](#component-guides)** - In-depth guides for each component
- **[Use Case Examples](#use-case-examples)** - Real-world implementation patterns

### 🔧 API Reference
- **[Core Components](api/core.md)** - Documents, Queries, and fundamental data structures
- **[Embedding Models](api/embedding_models.md)** - OpenAI, SentenceTransformers, and custom models
- **[Document Stores](api/document_stores.md)** - Vector databases and storage solutions
- **[Chunking Strategies](api/chunkers.md)** - Text chunking for optimal retrieval
- **[Retrievers](api/retrievers.md)** - Semantic and hybrid retrieval methods
- **[Rankers](api/rankers.md)** - Document ranking and reordering
- **[Generators](api/generators.md)** - LLM integration and response generation
- **[Pipelines](api/pipelines.md)** - End-to-end workflow orchestration

### 🛠️ Development
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute to RAG-Lib
- **[Custom Components](#custom-components)** - Building your own components
- **[Testing Guide](#testing)** - Testing strategies and best practices

## 🏃‍♂️ Quick Navigation

### I'm New to RAG
Start here to understand RAG concepts and build your first system:

1. **[What is RAG?](getting_started.md#what-is-rag)** - Understand the basics
2. **[Installation](../INSTALL.md)** - Set up your environment
3. **[First RAG System](getting_started.md#your-first-rag-system)** - 5-minute tutorial
4. **[Basic Pipeline Tutorial](tutorials/basic_rag_pipeline.md)** - Complete implementation

### I Want to Build Production Systems
Advanced guides for production deployment:

1. **[Configuration Management](configuration.md)** - Environment-specific configs
2. **[Production RAG Pipeline](tutorials/basic_rag_pipeline.md#production-considerations)** - Scalable architecture
3. **[Performance Optimization](#performance-optimization)** - Speed and memory optimization
4. **[Monitoring & Debugging](#monitoring)** - Production monitoring

### I Want to Customize Components
Build custom components for your specific needs:

1. **[Custom Embedding Models](#custom-embedding-models)** - Integrate new models
2. **[Custom Document Stores](#custom-document-stores)** - Add new vector databases
3. **[Custom Chunkers](#custom-chunkers)** - Implement domain-specific chunking
4. **[Contributing](../CONTRIBUTING.md)** - Contribute back to the project

## 🧩 Component Guides

### Embedding Models
Transform text into semantic vectors for similarity search.

**Available Models:**
- **OpenAI Embeddings**: High-quality cloud-based embeddings
- **SentenceTransformers**: Local, privacy-preserving models
- **Custom Models**: Integrate your own embedding models

**Quick Start:**
```python
from rag_lib import OpenAIEmbedding
embedding_model = OpenAIEmbedding("text-embedding-3-small")
```

**[📖 Full Guide](api/embedding_models.md)**

### Document Stores
Store and efficiently retrieve documents with their embeddings.

**Available Stores:**
- **InMemoryDocumentStore**: Fast, perfect for development
- **QdrantDocumentStore**: Scalable vector database
- **Custom Stores**: Add your preferred vector database

**Quick Start:**
```python
from rag_lib import InMemoryDocumentStore
store = InMemoryDocumentStore(embedding_model=embedding_model)
```

**[📖 Full Guide](api/document_stores.md)**

### Chunking Strategies
Split documents into optimal chunks for better retrieval.

**Available Chunkers:**
- **Length-based**: Simple token-based chunking
- **Recursive**: Hierarchical chunking preserving structure
- **Semantic**: AI-powered chunking based on meaning

**Quick Start:**
```python
from rag_lib.chunkers import create_chunker
chunker = create_chunker("recursive", {"max_tokens": 1024})
```

**[📖 Full Guide](api/chunkers.md)**

### Retrievers
Find the most relevant documents for any query.

**Available Retrievers:**
- **SemanticRetriever**: Vector similarity search
- **HybridRetriever**: Combines multiple retrieval methods
- **Custom Retrievers**: Implement domain-specific retrieval

**Quick Start:**
```python
from rag_lib import SemanticRetriever
retriever = SemanticRetriever("main", store, embedding_model)
```

**[📖 Full Guide](api/retrievers.md)**

### Rankers
Reorder retrieved documents by relevance.

**Available Rankers:**
- **NoOpRanker**: Pass-through for simple cases
- **CrossEncoderRanker**: Neural reranking for better quality
- **Custom Rankers**: Domain-specific ranking algorithms

**Quick Start:**
```python
from rag_lib.rankers import CrossEncoderRanker
ranker = CrossEncoderRanker("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

**[📖 Full Guide](api/rankers.md)**

### Generators
Generate responses based on retrieved context.

**Available Generators:**
- **OpenAI Generator**: GPT-3.5/4 integration
- **HuggingFace Generator**: Local and cloud models
- **Custom Generators**: Integrate any LLM

**Quick Start:**
```python
from your_generators import OpenAIGenerator
generator = OpenAIGenerator("gpt-3.5-turbo")
```

**[📖 Full Guide](api/generators.md)**

## 📋 Use Case Examples

### Academic Research
Process and query research papers, theses, and academic documents.

```python
# Optimized for academic content
config = {
    "chunker": {
        "type": "recursive_academic",
        "preserve_citations": True,
        "create_parent_chunks": True
    },
    "retriever": {"top_k": 20},
    "ranker": {"top_k": 8}
}
```

**[📖 Complete Example](examples/academic_research.md)**

### Enterprise Knowledge Base
Build internal knowledge systems for companies.

```python
# Enterprise-grade configuration
config = {
    "document_store": {"type": "qdrant"},
    "chunker": {"type": "recursive"},
    "security": {"enable_access_control": True}
}
```

**[📖 Complete Example](examples/enterprise_kb.md)**

### Customer Support
Create intelligent customer support systems.

```python
# Optimized for quick, accurate responses
config = {
    "retriever": {"top_k": 10, "fast_retrieval": True},
    "ranker": {"type": "cross_encoder", "top_k": 3},
    "generator": {"temperature": 0.1, "max_tokens": 200}
}
```

**[📖 Complete Example](examples/customer_support.md)**

### Code Documentation
Process and query software documentation and code.

```python
# Code-aware configuration
config = {
    "chunker": {
        "type": "recursive_code",
        "language_specific": True,
        "preserve_code_blocks": True
    }
}
```

**[📖 Complete Example](examples/code_documentation.md)**

## 🚀 Performance Optimization

### Memory Optimization
```python
# For memory-constrained environments
config = {
    "embedding_model": {"model_name": "all-MiniLM-L6-v2"},  # Smaller model
    "chunker": {"max_tokens": 512},  # Smaller chunks
    "batch_size": 5,  # Smaller batches
    "enable_streaming": True
}
```

### Speed Optimization
```python
# For high-performance requirements
config = {
    "embedding_model": {"batch_size": 100, "device": "cuda"},
    "document_store": {"type": "qdrant", "parallel_indexing": True},
    "retriever": {"enable_caching": True},
    "batch_size": 50
}
```

### Quality Optimization
```python
# For maximum quality
config = {
    "embedding_model": {"model_name": "text-embedding-3-large"},
    "chunker": {"type": "semantic", "similarity_threshold": 0.8},
    "retriever": {"top_k": 30},
    "ranker": {"type": "cross_encoder", "top_k": 10}
}
```

## 🏗️ Custom Components

### Custom Embedding Model
```python
from rag_lib.models.embedding import EmbeddingModel

class CustomEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str):
        super().__init__(model_name, dimension=768)
    
    def encode_text(self, text: str) -> List[float]:
        # Your custom encoding logic
        return embedding
```

### Custom Document Store
```python
from rag_lib.document_stores import BaseDocumentStore

class CustomDocumentStore(BaseDocumentStore):
    def create_index(self, index_name: str, config=None) -> bool:
        # Your custom index creation
        pass
    
    def add_documents(self, index_name: str, documents: List[Document]) -> bool:
        # Your custom document addition
        pass
```

### Custom Chunker
```python
from rag_lib.chunkers import BaseChunker

class CustomChunker(BaseChunker):
    def chunk_text(self, text: str) -> List[str]:
        # Your custom chunking logic
        return chunks
```

**[📖 Full Custom Components Guide](tutorials/custom_components.md)**

## 🧪 Testing

### Unit Testing
```python
import pytest
from rag_lib import Query, Document

def test_query_creation():
    query = Query(content="test query")
    assert query.content == "test query"
    assert "query_id" in query.metadata
```

### Integration Testing
```python
def test_rag_pipeline():
    # Test complete pipeline
    pipeline = create_test_pipeline()
    response = pipeline.query("test question", "test_index")
    assert response.generated_text
    assert len(response.retrieved_documents) > 0
```

### Component Testing
```python
# Run the comprehensive test suite
python examples/test_rag_components.py
```

**[📖 Full Testing Guide](testing.md)**

## 📊 Monitoring

### Performance Metrics
```python
# Built-in timing
response = pipeline.query(query, index_name)
print(f"Retrieval: {response.retrieval_time:.2f}s")
print(f"Generation: {response.generation_time:.2f}s")
print(f"Total: {response.total_time:.2f}s")
```

### Quality Metrics
```python
# Relevance scoring
for doc in response.retrieved_documents:
    print(f"Document: {doc.document.id}, Score: {doc.score:.3f}")

# Response quality indicators
quality_score = calculate_response_quality(response)
```

### Error Tracking
```python
import logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag-lib.log'),
        logging.StreamHandler()
    ]
)
```

## 🆘 Troubleshooting

### Common Issues

#### Installation Problems
- **Missing dependencies**: Check [INSTALL.md](../INSTALL.md#troubleshooting)
- **Version conflicts**: Use virtual environments
- **Permission errors**: Check file permissions

#### Performance Issues
- **Slow queries**: Reduce chunk size and batch size
- **High memory usage**: Use smaller embedding models
- **Poor relevance**: Experiment with chunking strategies

#### API Errors
- **Rate limiting**: Implement exponential backoff
- **Authentication**: Verify API keys in environment variables
- **Timeouts**: Increase timeout settings

### Getting Help

1. **Check Documentation**: Search this documentation first
2. **Review Examples**: Look at working examples in `examples/`
3. **GitHub Issues**: [Create an issue](https://github.com/tungnk99/rag-lib/issues) for bugs
4. **Discussions**: [Join discussions](https://github.com/tungnk99/rag-lib/discussions) for questions
5. **Email Support**: tungnk.hust@gmail.com for direct help

## 🎯 Best Practices

### Development
- Start with simple configurations and gradually add complexity
- Use version control for configuration files
- Test with small datasets before scaling up
- Monitor performance metrics from the beginning

### Production
- Use environment-specific configurations
- Implement comprehensive error handling
- Set up monitoring and alerting
- Plan for scaling and performance optimization

### Security
- Never commit API keys to version control
- Use environment variables for sensitive data
- Implement access controls for enterprise deployments
- Regular security audits and updates

## 🔄 Migration Guides

### From Other RAG Libraries
- **From LangChain**: [Migration guide](migration/from_langchain.md)
- **From Haystack**: [Migration guide](migration/from_haystack.md)
- **From Custom Solutions**: [Migration guide](migration/from_custom.md)

### Version Upgrades
- **v1.0 to v1.1**: [Upgrade guide](migration/v1.0_to_v1.1.md)
- **Breaking Changes**: [Change log](../CHANGELOG.md)

## 📅 Roadmap

### Current Version (v1.0)
- ✅ Core RAG pipeline components
- ✅ Multiple embedding models and document stores
- ✅ Comprehensive configuration system
- ✅ Rich examples and documentation

### Upcoming Features (v1.1)
- 🔄 Async/await support for better performance
- 🔄 More vector database integrations (Pinecone, Weaviate)
- 🔄 Advanced retrieval methods (hybrid search, reranking)
- 🔄 Conversation history and memory

### Future Plans (v2.0)
- 🔮 Multi-modal RAG (text, images, audio)
- 🔮 Federated search across multiple sources
- 🔮 Real-time learning and adaptation
- 🔮 Enterprise security and compliance features

## 🤝 Community

### Contributing
We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Code contributions
- Documentation improvements
- Bug reports and feature requests
- Community support

### Community Resources
- **GitHub**: [RAG-Lib Repository](https://github.com/tungnk99/rag-lib)
- **Discussions**: [GitHub Discussions](https://github.com/tungnk99/rag-lib/discussions)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/tungnk99/rag-lib/issues)
- **Email**: tungnk.hust@gmail.com

---

## 📜 License

RAG-Lib is released under the [MIT License](../LICENSE). You're free to use, modify, and distribute it for both commercial and non-commercial purposes.

---

**Happy building with RAG-Lib! 🚀**

*Last updated: January 2024*
