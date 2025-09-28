# Getting Started with RAG-Lib

Welcome to RAG-Lib! This guide will help you build your first RAG (Retrieval-Augmented Generation) system in just a few minutes.

## Table of Contents
- [What is RAG?](#what-is-rag)
- [Installation](#installation)
- [Your First RAG System](#your-first-rag-system)
- [Understanding the Components](#understanding-the-components)
- [Next Steps](#next-steps)

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines:
1. **Document Retrieval**: Finding relevant information from a knowledge base
2. **Text Generation**: Using that information to generate accurate responses

RAG-Lib makes it easy to build production-ready RAG systems with minimal code.

## Installation

### Quick Install
```bash
# Clone the repository
git clone https://github.com/tungnk99/rag-lib.git
cd rag-lib

# Install with basic dependencies
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

### Environment Setup
```bash
# Create .env file for API keys
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# For local embeddings only (no API key needed)
pip install -e ".[sbert]"
```

See [INSTALL.md](../INSTALL.md) for detailed installation instructions.

## Your First RAG System

Let's build a simple RAG system that can answer questions about your documents:

### Step 1: Basic Setup

```python
# first_rag_system.py
import os
from rag_lib import (
    Query, Document, InMemoryDocumentStore, 
    SemanticRetriever, NoOpRanker
)

# For this example, we'll use a local embedding model
from rag_lib.models.embedding.sbert_embedding import SBERTEmbedding

def create_basic_rag():
    """Create a basic RAG system with local components."""
    
    # 1. Initialize embedding model (no API key needed)
    print("🚀 Initializing embedding model...")
    embedding_model = SBERTEmbedding(
        model_name="all-MiniLM-L6-v2"  # Fast, lightweight model
    )
    
    # 2. Create document store
    print("📚 Setting up document store...")
    document_store = InMemoryDocumentStore(
        embedding_model=embedding_model
    )
    
    # 3. Create index for our documents
    document_store.create_index(
        "knowledge_base", 
        vector_size=embedding_model.dimension
    )
    
    # 4. Create retriever
    print("🔍 Setting up retriever...")
    retriever = SemanticRetriever(
        name="main_retriever",
        document_store=document_store,
        embedding_model=embedding_model
    )
    
    # 5. Create ranker (optional, but good practice)
    ranker = NoOpRanker()  # Simple pass-through ranker
    
    return retriever, ranker, document_store

if __name__ == "__main__":
    retriever, ranker, document_store = create_basic_rag()
    print("✅ RAG system initialized successfully!")
```

### Step 2: Add Your Documents

```python
# Continue in the same file or create add_documents.py

def add_sample_documents(document_store):
    """Add some sample documents to the knowledge base."""
    
    # Sample documents about technology
    documents = [
        Document(
            id="python_intro",
            content="Python is a high-level programming language known for its "
                   "simplicity and readability. It's widely used in web development, "
                   "data science, artificial intelligence, and automation.",
            metadata={"category": "programming", "difficulty": "beginner"}
        ),
        Document(
            id="ml_basics",
            content="Machine Learning is a subset of artificial intelligence that "
                   "enables computers to learn and make decisions from data without "
                   "being explicitly programmed. It includes supervised, unsupervised, "
                   "and reinforcement learning.",
            metadata={"category": "ai", "difficulty": "intermediate"}
        ),
        Document(
            id="rag_explanation",
            content="Retrieval-Augmented Generation (RAG) combines information "
                   "retrieval with text generation. It first searches a knowledge base "
                   "for relevant information, then uses that context to generate "
                   "accurate and informed responses.",
            metadata={"category": "ai", "difficulty": "advanced"}
        ),
        Document(
            id="vector_db",
            content="Vector databases store high-dimensional vectors that represent "
                   "data embeddings. They enable fast similarity search and are "
                   "essential for applications like semantic search, recommendation "
                   "systems, and RAG pipelines.",
            metadata={"category": "database", "difficulty": "intermediate"}
        )
    ]
    
    print(f"📄 Adding {len(documents)} documents to knowledge base...")
    
    # Add documents to the store (embeddings will be computed automatically)
    document_store.add_documents("knowledge_base", documents)
    
    print("✅ Documents added successfully!")
    return documents

# Add this to your main function
if __name__ == "__main__":
    retriever, ranker, document_store = create_basic_rag()
    documents = add_sample_documents(document_store)
```

### Step 3: Query Your RAG System

```python
# Continue adding to your file

def query_rag_system(retriever, ranker):
    """Query the RAG system with sample questions."""
    
    # Sample queries
    queries = [
        "What is Python programming?",
        "How does machine learning work?", 
        "Explain RAG systems",
        "What are vector databases used for?"
    ]
    
    print("\n🔍 Querying the RAG system...")
    print("=" * 50)
    
    for query_text in queries:
        print(f"\n❓ Question: {query_text}")
        
        # Create query object
        query = Query(content=query_text)
        
        # Retrieve relevant documents
        retrieval_results = retriever.retrieve(
            query=query,
            index_name="knowledge_base",
            top_k=3  # Get top 3 most relevant documents
        )
        
        # Rank results (NoOpRanker just passes them through)
        ranked_results = ranker.rerank(
            documents=[r.document for r in retrieval_results],
            query=query,
            top_k=2
        )
        
        print(f"📚 Found {len(ranked_results)} relevant documents:")
        
        for i, result in enumerate(ranked_results, 1):
            # Get the original retrieval result to access the score
            original_result = next(
                r for r in retrieval_results 
                if r.document.id == result.id
            )
            
            print(f"\n  {i}. Document: {result.id}")
            print(f"     Score: {original_result.score:.3f}")
            print(f"     Content: {result.content[:100]}...")
            print(f"     Category: {result.metadata.get('category', 'N/A')}")

# Add this to your main function
if __name__ == "__main__":
    retriever, ranker, document_store = create_basic_rag()
    documents = add_sample_documents(document_store)
    query_rag_system(retriever, ranker)
```

### Step 4: Complete Example

Here's the complete working example:

```python
# complete_first_rag.py
import os
from rag_lib import (
    Query, Document, InMemoryDocumentStore, 
    SemanticRetriever, NoOpRanker
)
from rag_lib.models.embedding.sbert_embedding import SBERTEmbedding

def main():
    """Complete RAG system example."""
    
    print("🚀 Building your first RAG system with RAG-Lib!")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n1️⃣ Initializing embedding model...")
    embedding_model = SBERTEmbedding(model_name="all-MiniLM-L6-v2")
    print(f"   ✅ Model loaded: {embedding_model.model_name}")
    print(f"   📐 Embedding dimension: {embedding_model.dimension}")
    
    print("\n2️⃣ Setting up document store...")
    document_store = InMemoryDocumentStore(embedding_model=embedding_model)
    document_store.create_index("knowledge_base", vector_size=embedding_model.dimension)
    print("   ✅ Document store ready")
    
    print("\n3️⃣ Creating retriever...")
    retriever = SemanticRetriever(
        name="main_retriever",
        document_store=document_store,
        embedding_model=embedding_model
    )
    ranker = NoOpRanker()
    print("   ✅ Retriever and ranker ready")
    
    # 2. Add documents
    print("\n4️⃣ Adding sample documents...")
    documents = [
        Document(
            id="python",
            content="Python is a programming language that's easy to learn and powerful. "
                   "It's used for web development, data science, AI, and automation.",
            metadata={"topic": "programming"}
        ),
        Document(
            id="machine_learning", 
            content="Machine learning enables computers to learn from data without "
                   "explicit programming. It's used in recommendation systems, "
                   "image recognition, and natural language processing.",
            metadata={"topic": "ai"}
        ),
        Document(
            id="rag_systems",
            content="RAG systems combine document retrieval with text generation. "
                   "They search for relevant information and use it to generate "
                   "accurate, contextual responses.",
            metadata={"topic": "ai"}
        )
    ]
    
    document_store.add_documents("knowledge_base", documents)
    print(f"   ✅ Added {len(documents)} documents")
    
    # 3. Query the system
    print("\n5️⃣ Testing queries...")
    queries = [
        "What is Python used for?",
        "How does machine learning work?",
        "Explain RAG systems"
    ]
    
    for i, query_text in enumerate(queries, 1):
        print(f"\n   Query {i}: {query_text}")
        
        # Create and process query
        query = Query(content=query_text)
        results = retriever.retrieve(query, "knowledge_base", top_k=2)
        
        print(f"   📄 Retrieved {len(results)} documents:")
        for j, result in enumerate(results, 1):
            print(f"      {j}. {result.document.id} (score: {result.score:.3f})")
            print(f"         {result.document.content[:80]}...")
    
    print("\n🎉 Your first RAG system is working!")
    print("\n💡 Next steps:")
    print("   - Add your own documents")
    print("   - Try different embedding models") 
    print("   - Experiment with chunking strategies")
    print("   - Add a language model for generation")

if __name__ == "__main__":
    main()
```

### Run Your First RAG System

```bash
# Save the code as complete_first_rag.py and run:
python complete_first_rag.py
```

You should see output like:
```
🚀 Building your first RAG system with RAG-Lib!
==================================================

1️⃣ Initializing embedding model...
   ✅ Model loaded: all-MiniLM-L6-v2
   📐 Embedding dimension: 384

2️⃣ Setting up document store...
   ✅ Document store ready

3️⃣ Creating retriever...
   ✅ Retriever and ranker ready

4️⃣ Adding sample documents...
   ✅ Added 3 documents

5️⃣ Testing queries...
   Query 1: What is Python used for?
   📄 Retrieved 2 documents:
      1. python (score: 0.742)
         Python is a programming language that's easy to learn and powerful...

🎉 Your first RAG system is working!
```

## Understanding the Components

### 🧠 Embedding Model
Converts text into numerical vectors that capture semantic meaning:

```python
# Local model (free, runs on your computer)
from rag_lib.models.embedding.sbert_embedding import SBERTEmbedding
embedding_model = SBERTEmbedding("all-MiniLM-L6-v2")

# OpenAI model (requires API key, higher quality)
from rag_lib import OpenAIEmbedding
embedding_model = OpenAIEmbedding("text-embedding-3-small")
```

### 📚 Document Store
Stores documents and their embeddings for fast retrieval:

```python
# In-memory store (good for development/small datasets)
from rag_lib import InMemoryDocumentStore
store = InMemoryDocumentStore(embedding_model=embedding_model)

# Qdrant store (good for production/large datasets)
from rag_lib.document_stores import QdrantDocumentStore
store = QdrantDocumentStore(url="http://localhost:6333")
```

### 🔍 Retriever
Finds relevant documents based on semantic similarity:

```python
from rag_lib import SemanticRetriever

retriever = SemanticRetriever(
    name="my_retriever",
    document_store=store,
    embedding_model=embedding_model
)

# Advanced usage with filters
results = retriever.retrieve(
    query="machine learning",
    index_name="docs",
    top_k=5,
    filters={"category": "ai"}  # Only search AI documents
)
```

### 🏆 Ranker
Reorders retrieved documents by relevance (optional but recommended):

```python
from rag_lib.rankers import NoOpRanker, CrossEncoderRanker

# Simple pass-through ranker
ranker = NoOpRanker()

# Advanced neural ranker (requires more setup)
ranker = CrossEncoderRanker("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

## Next Steps

Now that you have a working RAG system, here are some next steps:

### 1. Add Real Documents
Replace the sample documents with your own:

```python
# Read documents from files
from rag_lib.readers import get_reader_for_file

reader = get_reader_for_file("your_document.pdf")
file_content = reader.read("your_document.pdf")

# Convert to Document objects
documents = []
for i, element in enumerate(file_content.elements):
    if element.type == ElementType.TEXT:
        doc = Document(
            id=f"doc_{i}",
            content=element.content,
            metadata={"source": file_content.file_path}
        )
        documents.append(doc)

# Add to store
document_store.add_documents("knowledge_base", documents)
```

### 2. Use Document Processing Pipelines
For larger document collections:

```python
from rag_lib.pipelines.data_pipeline import create_pipeline_from_config

config = {
    "chunker": {
        "type": "recursive",
        "max_tokens": 512,
        "overlap_tokens": 50
    },
    "batch_size": 10
}

pipeline = create_pipeline_from_config(config, document_store)
results = pipeline.process_files(
    file_paths=["doc1.pdf", "doc2.docx", "doc3.txt"],
    index_name="my_docs"
)
```

### 3. Add Text Generation
Complete your RAG system with a generator:

```python
from rag_lib.pipelines import RAGPipeline
from your_generator import YourGenerator  # You'll need to implement this

generator = YourGenerator()  # Using OpenAI, Hugging Face, etc.

rag_pipeline = RAGPipeline(
    retriever=retriever,
    ranker=ranker,
    generator=generator
)

response = rag_pipeline.query(
    query="What is machine learning?",
    index_name="knowledge_base"
)

print(f"Answer: {response.generated_text}")
```

### 4. Explore Advanced Features

- **[Semantic Chunking](tutorials/semantic_chunking.md)**: Better text splitting
- **[Configuration Management](tutorials/configuration.md)**: Environment-specific configs
- **[Custom Components](tutorials/custom_components.md)**: Build your own components
- **[Production Deployment](tutorials/production_rag.md)**: Scale for production

### 5. Check Out Examples

Look at the `examples/` directory for more advanced use cases:
- [Data Pipeline Example](../examples/data_pipeline_config_example.py)
- [Semantic Chunking Example](../examples/semantic_chunker_example.py)
- [Cross-Encoder Ranking](../examples/cross_encoder_ranker_example.py)

## Common Issues & Solutions

### Issue: "Model not found" Error
```bash
# Solution: Models download on first use
# Wait for download to complete or check internet connection
```

### Issue: Memory Issues with Large Documents
```python
# Solution: Use smaller chunk sizes
config = {
    "chunker": {
        "type": "length",
        "max_tokens": 256,  # Smaller chunks
        "overlap_tokens": 25
    },
    "batch_size": 5  # Smaller batches
}
```

### Issue: Poor Retrieval Quality
```python
# Solution: Try different embedding models
# For better quality (but slower):
embedding_model = SBERTEmbedding("all-mpnet-base-v2")

# For multilingual support:
embedding_model = SBERTEmbedding("paraphrase-multilingual-MiniLM-L12-v2")
```

## Getting Help

- **Documentation**: Check the [docs/](../) directory
- **Examples**: Look at working examples in [examples/](../examples/)
- **Issues**: Create a [GitHub issue](https://github.com/tungnk99/rag-lib/issues)
- **Discussions**: Join [GitHub discussions](https://github.com/tungnk99/rag-lib/discussions)

Happy building! 🚀
