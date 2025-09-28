# Building a Complete RAG Pipeline

This tutorial walks you through building a complete RAG (Retrieval-Augmented Generation) pipeline that can process documents, answer questions, and generate responses.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Document Processing](#document-processing)
- [Building the RAG Pipeline](#building-the-rag-pipeline)
- [Advanced Features](#advanced-features)
- [Production Considerations](#production-considerations)

## Overview

A complete RAG pipeline consists of:

1. **Document Processing**: Reading and chunking documents
2. **Embedding**: Converting text to vectors
3. **Storage**: Storing documents and embeddings
4. **Retrieval**: Finding relevant documents
5. **Ranking**: Ordering results by relevance
6. **Generation**: Creating responses based on retrieved context

## Setup

### Prerequisites
```bash
# Install RAG-Lib with all dependencies
pip install -e ".[all]"

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"  # If using OpenAI
```

### Project Structure
```
my_rag_project/
├── documents/           # Your document files
├── configs/            # Configuration files
├── data/              # Processed data
├── main.py            # Main pipeline script
├── config.json        # Pipeline configuration
└── .env              # Environment variables
```

## Document Processing

### Step 1: Prepare Your Documents

```python
# document_processor.py
import os
from pathlib import Path
from typing import List

from rag_lib.readers import create_multi_format_reader
from rag_lib.schemas.schema import Document, ElementType

def collect_documents(documents_dir: str) -> List[str]:
    """Collect all supported document files from directory."""
    supported_extensions = {'.pdf', '.docx', '.txt', '.xlsx'}
    
    file_paths = []
    for root, dirs, files in os.walk(documents_dir):
        for file in files:
            if Path(file).suffix.lower() in supported_extensions:
                file_paths.append(os.path.join(root, file))
    
    return file_paths

def process_documents(file_paths: List[str]) -> List[Document]:
    """Process documents and extract content."""
    reader = create_multi_format_reader()
    documents = []
    
    for file_path in file_paths:
        try:
            print(f"📄 Processing: {file_path}")
            
            # Read document
            file_content = reader.read(file_path)
            
            # Extract text elements and create documents
            for i, element in enumerate(file_content.elements):
                if element.type == ElementType.TEXT and element.content.strip():
                    doc = Document(
                        id=f"{Path(file_path).stem}_{i}",
                        content=element.content,
                        metadata={
                            "source_file": file_path,
                            "element_index": i,
                            "file_type": file_content.metadata.get("file_type"),
                            "page_number": element.metadata.get("page_number", 1)
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    return documents

if __name__ == "__main__":
    # Example usage
    docs_dir = "documents/"
    file_paths = collect_documents(docs_dir)
    print(f"Found {len(file_paths)} documents")
    
    documents = process_documents(file_paths)
    print(f"Extracted {len(documents)} text segments")
```

### Step 2: Smart Document Chunking

```python
# chunking_strategy.py
from rag_lib.chunkers import create_chunker
from rag_lib.schemas.schema import Document
from typing import List

def chunk_documents(documents: List[Document], chunker_config: dict) -> List[Document]:
    """Chunk documents using specified strategy."""
    
    # Create chunker based on config
    chunker = create_chunker(
        chunker_type=chunker_config["type"],
        config=chunker_config
    )
    
    chunked_documents = []
    
    for doc in documents:
        try:
            # Chunk the document content
            chunks = chunker.chunk_text(doc.content)
            
            # Create new documents for each chunk
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "parent_id": doc.id,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                chunked_documents.append(chunk_doc)
                
        except Exception as e:
            print(f"❌ Error chunking document {doc.id}: {e}")
            continue
    
    return chunked_documents

# Example configurations for different use cases
CHUNKING_CONFIGS = {
    "general": {
        "type": "length",
        "max_tokens": 512,
        "overlap_tokens": 50,
        "preserve_sentences": True
    },
    "academic": {
        "type": "recursive_academic",
        "overlap_tokens": 75,
        "create_parent_chunks": True
    },
    "code": {
        "type": "recursive_code", 
        "overlap_tokens": 25,
        "create_parent_chunks": True
    },
    "semantic": {
        "type": "semantic_sentence_transformer",
        "model_name": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.7,
        "max_tokens": 512,
        "min_tokens": 50
    }
}
```

## Building the RAG Pipeline

### Step 3: Configure Your Pipeline

```json
// config.json
{
  "embedding_model": {
    "type": "openai",
    "model_name": "text-embedding-3-small",
    "dimension": 1536
  },
  "document_store": {
    "type": "inmemory",
    "config": {}
  },
  "chunker": {
    "type": "recursive",
    "max_tokens": 1024,
    "overlap_tokens": 100,
    "create_parent_chunks": true
  },
  "retriever": {
    "type": "semantic",
    "top_k": 10,
    "similarity_threshold": 0.7
  },
  "ranker": {
    "type": "cross_encoder",
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_k": 5
  },
  "generator": {
    "type": "openai",
    "model_name": "gpt-3.5-turbo",
    "max_tokens": 500,
    "temperature": 0.1
  }
}
```

### Step 4: Complete Pipeline Implementation

```python
# rag_pipeline.py
import json
import time
from typing import List, Dict, Any

from rag_lib import (
    OpenAIEmbedding, InMemoryDocumentStore, SemanticRetriever,
    Query, Document, RAGPipeline
)
from rag_lib.rankers import CrossEncoderRanker, NoOpRanker
from rag_lib.generators import BaseGenerator, GenerationResult
from rag_lib.models.llm import OpenAILLM

class OpenAIGenerator(BaseGenerator):
    """Generator using OpenAI LLM."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        self.llm = OpenAILLM(model_name=model_name, **kwargs)
    
    def generate(self, query: Query, documents: List[Document], config=None):
        """Generate response based on retrieved documents."""
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):  # Limit to top 5
            context_parts.append(f"Document {i}:\n{doc.content}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context documents, answer the question.

Context:
{context}

Question: {query.content}

Answer: Provide a comprehensive answer based on the context. If the context doesn't contain enough information, say so."""

        # Generate response
        try:
            response = self.llm.completion(
                prompt=prompt,
                max_tokens=config.get("max_tokens", 500) if config else 500,
                temperature=config.get("temperature", 0.1) if config else 0.1
            )
            
            return GenerationResult(
                generated_text=response.text,
                metadata={
                    "model": self.llm.model_name,
                    "documents_used": len(documents),
                    "prompt_tokens": response.usage.get("prompt_tokens", 0),
                    "completion_tokens": response.usage.get("completion_tokens", 0)
                }
            )
            
        except Exception as e:
            return GenerationResult(
                generated_text=f"Error generating response: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def is_available(self) -> bool:
        """Check if generator is available."""
        return True

class CompletePipeline:
    """Complete RAG pipeline with all components."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize pipeline from configuration."""
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self._setup_components()
    
    def _setup_components(self):
        """Setup all pipeline components."""
        
        print("🚀 Initializing RAG Pipeline...")
        
        # 1. Embedding Model
        print("🧠 Setting up embedding model...")
        embedding_config = self.config["embedding_model"]
        
        if embedding_config["type"] == "openai":
            self.embedding_model = OpenAIEmbedding(
                model_name=embedding_config["model_name"]
            )
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_config['type']}")
        
        # 2. Document Store
        print("📚 Setting up document store...")
        store_config = self.config["document_store"]
        
        if store_config["type"] == "inmemory":
            self.document_store = InMemoryDocumentStore(
                embedding_model=self.embedding_model
            )
        else:
            raise ValueError(f"Unsupported store type: {store_config['type']}")
        
        # 3. Retriever
        print("🔍 Setting up retriever...")
        self.retriever = SemanticRetriever(
            name="main_retriever",
            document_store=self.document_store,
            embedding_model=self.embedding_model
        )
        
        # 4. Ranker
        print("🏆 Setting up ranker...")
        ranker_config = self.config["ranker"]
        
        if ranker_config["type"] == "cross_encoder":
            self.ranker = CrossEncoderRanker(
                model_name=ranker_config["model_name"]
            )
        else:
            self.ranker = NoOpRanker()
        
        # 5. Generator
        print("✨ Setting up generator...")
        generator_config = self.config["generator"]
        
        if generator_config["type"] == "openai":
            self.generator = OpenAIGenerator(
                model_name=generator_config["model_name"],
                max_tokens=generator_config.get("max_tokens", 500),
                temperature=generator_config.get("temperature", 0.1)
            )
        else:
            raise ValueError(f"Unsupported generator type: {generator_config['type']}")
        
        # 6. RAG Pipeline
        print("🔗 Assembling RAG pipeline...")
        self.rag_pipeline = RAGPipeline(
            retriever=self.retriever,
            ranker=self.ranker,
            generator=self.generator
        )
        
        print("✅ Pipeline initialization complete!")
    
    def load_documents(self, documents: List[Document], index_name: str = "main"):
        """Load documents into the pipeline."""
        
        print(f"📄 Loading {len(documents)} documents...")
        
        # Create index
        self.document_store.create_index(
            index_name, 
            vector_size=self.embedding_model.dimension
        )
        
        # Add documents in batches for better performance
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.document_store.add_documents(index_name, batch)
            print(f"   Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
        
        print("✅ Documents loaded successfully!")
    
    def query(self, question: str, index_name: str = "main") -> Dict[str, Any]:
        """Query the RAG pipeline."""
        
        start_time = time.time()
        
        # Create query object
        query = Query(content=question)
        
        # Get response from pipeline
        response = self.rag_pipeline.query(
            query=query,
            index_name=index_name,
            top_k=self.config["retriever"]["top_k"],
            rerank_top_k=self.config["ranker"].get("top_k", 5)
        )
        
        # Format results
        result = {
            "question": question,
            "answer": response.generated_text,
            "retrieved_documents": [
                {
                    "id": doc.document.id,
                    "score": doc.score,
                    "content": doc.document.content[:200] + "..." if len(doc.document.content) > 200 else doc.document.content,
                    "metadata": doc.document.metadata
                }
                for doc in response.retrieved_documents
            ],
            "metadata": {
                "total_time": time.time() - start_time,
                "retrieval_time": response.retrieval_time,
                "generation_time": response.generation_time,
                "documents_retrieved": len(response.retrieved_documents)
            }
        }
        
        return result
    
    def batch_query(self, questions: List[str], index_name: str = "main") -> List[Dict[str, Any]]:
        """Process multiple queries efficiently."""
        
        print(f"🔄 Processing {len(questions)} queries...")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"   Query {i}/{len(questions)}: {question[:50]}...")
            
            result = self.query(question, index_name)
            results.append(result)
        
        return results

def main():
    """Example usage of complete pipeline."""
    
    # Initialize pipeline
    pipeline = CompletePipeline("config.json")
    
    # Load sample documents (replace with your document processing)
    from document_processor import collect_documents, process_documents
    from chunking_strategy import chunk_documents, CHUNKING_CONFIGS
    
    # Process documents
    file_paths = collect_documents("documents/")
    documents = process_documents(file_paths)
    
    # Chunk documents
    chunked_docs = chunk_documents(
        documents, 
        CHUNKING_CONFIGS["general"]
    )
    
    # Load into pipeline
    pipeline.load_documents(chunked_docs)
    
    # Example queries
    questions = [
        "What are the main topics covered in the documents?",
        "Can you explain the key concepts?",
        "What are the important findings or conclusions?"
    ]
    
    # Query the system
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 50)
        
        result = pipeline.query(question)
        
        print(f"💡 Answer: {result['answer']}")
        print(f"📊 Retrieved {result['metadata']['documents_retrieved']} documents")
        print(f"⏱️ Total time: {result['metadata']['total_time']:.2f}s")
        
        print("\n📚 Source documents:")
        for i, doc in enumerate(result['retrieved_documents'][:3], 1):
            print(f"   {i}. {doc['id']} (score: {doc['score']:.3f})")
            print(f"      {doc['content'][:100]}...")

if __name__ == "__main__":
    main()
```

## Advanced Features

### Semantic Chunking

For better chunk quality, use semantic chunking:

```python
# Advanced chunking with semantic similarity
chunker_config = {
    "type": "semantic_sentence_transformer",
    "model_name": "all-mpnet-base-v2",  # Higher quality model
    "similarity_threshold": 0.8,
    "max_tokens": 1024,
    "min_tokens": 100
}

chunked_docs = chunk_documents(documents, chunker_config)
```

### Multiple Index Management

Organize documents by topic or source:

```python
# Load different document types into separate indices
pipeline.load_documents(technical_docs, "technical")
pipeline.load_documents(business_docs, "business") 
pipeline.load_documents(legal_docs, "legal")

# Query specific indices
tech_result = pipeline.query("How does the API work?", "technical")
business_result = pipeline.query("What's our revenue model?", "business")
```

### Custom Filtering

Filter results by metadata:

```python
# Query with filters
from rag_lib.schemas.schema import Query

query = Query(
    content="machine learning algorithms",
    metadata={"filters": {"category": "ai", "year": 2024}}
)

results = pipeline.retriever.retrieve(
    query=query,
    index_name="main",
    top_k=10,
    filters={"category": "ai", "year": 2024}
)
```

## Production Considerations

### Performance Optimization

```python
# Optimize for production
production_config = {
    "embedding_model": {
        "type": "openai",
        "model_name": "text-embedding-3-small",  # Faster than large
        "batch_size": 100  # Batch API calls
    },
    "document_store": {
        "type": "qdrant",  # Use external vector DB
        "url": "http://localhost:6333",
        "collection_config": {
            "size": 1536,
            "distance": "Cosine"
        }
    },
    "retriever": {
        "top_k": 20,  # Retrieve more for reranking
        "similarity_threshold": 0.6
    },
    "ranker": {
        "type": "cross_encoder",
        "top_k": 5,  # Final results after reranking
        "batch_size": 32
    }
}
```

### Error Handling and Monitoring

```python
import logging
from typing import Optional

class ProductionPipeline(CompletePipeline):
    """Production-ready pipeline with monitoring."""
    
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(__name__)
        super().__init__(config_path)
    
    def query(self, question: str, index_name: str = "main") -> Optional[Dict[str, Any]]:
        """Query with comprehensive error handling."""
        
        try:
            self.logger.info(f"Processing query: {question[:100]}...")
            
            result = super().query(question, index_name)
            
            # Log metrics
            self.logger.info(
                f"Query successful - Time: {result['metadata']['total_time']:.2f}s, "
                f"Docs: {result['metadata']['documents_retrieved']}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}", exc_info=True)
            
            return {
                "question": question,
                "answer": "I apologize, but I encountered an error processing your query. Please try again.",
                "error": str(e),
                "metadata": {"error": True}
            }
```

### Caching Strategy

```python
import pickle
import hashlib
from functools import lru_cache

class CachedPipeline(ProductionPipeline):
    """Pipeline with result caching."""
    
    def __init__(self, config_path: str, cache_dir: str = "cache/"):
        super().__init__(config_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, question: str, index_name: str) -> str:
        """Generate cache key for query."""
        content = f"{question}:{index_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def query(self, question: str, index_name: str = "main") -> Dict[str, Any]:
        """Query with caching."""
        
        cache_key = self._get_cache_key(question, index_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                    result['metadata']['from_cache'] = True
                    self.logger.info(f"Cache hit for query: {question[:50]}...")
                    return result
            except Exception as e:
                self.logger.warning(f"Cache read error: {e}")
        
        # Process query
        result = super().query(question, index_name)
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
        
        return result
```

## Next Steps

1. **Experiment with Components**: Try different embedding models, chunking strategies, and rankers
2. **Add Evaluation**: Implement metrics to measure RAG quality
3. **Scale Up**: Move to production vector databases like Qdrant or Pinecone
4. **Add Features**: Implement conversation history, user feedback, and continuous learning
5. **Deploy**: Set up REST API, monitoring, and deployment infrastructure

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch sizes and chunk sizes
2. **Slow Performance**: Use smaller embedding models or external vector DBs
3. **Poor Results**: Experiment with different chunking strategies and similarity thresholds
4. **API Errors**: Implement retry logic and rate limiting

### Performance Monitoring

```python
# Add metrics collection
import time
from collections import defaultdict

class MetricsPipeline(ProductionPipeline):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.metrics = defaultdict(list)
    
    def query(self, question: str, index_name: str = "main") -> Dict[str, Any]:
        start_time = time.time()
        result = super().query(question, index_name)
        
        # Collect metrics
        self.metrics['query_time'].append(time.time() - start_time)
        self.metrics['docs_retrieved'].append(result['metadata']['documents_retrieved'])
        
        return result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        import statistics
        
        return {
            'avg_query_time': statistics.mean(self.metrics['query_time']),
            'avg_docs_retrieved': statistics.mean(self.metrics['docs_retrieved']),
            'total_queries': len(self.metrics['query_time'])
        }
```

This tutorial provides a solid foundation for building production-ready RAG systems. Start with the basic pipeline and gradually add advanced features as needed.
