# Installation Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Options](#installation-options)
- [Development Installation](#development-installation)
- [Optional Dependencies](#optional-dependencies)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Environment Setup](#environment-setup)

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM (8GB+ recommended for large documents)
- **Storage**: 2GB+ free space (more if using local embedding models)

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.9
```

#### Windows
1. Install Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. Install Microsoft Visual C++ Build Tools
3. Enable Windows Subsystem for Linux (WSL) - recommended

## Installation Options

### Option 1: Basic Installation (Recommended for Beginners)
```bash
# Clone the repository
git clone https://github.com/tungnk99/rag-lib.git
cd rag-lib

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install the package
pip install -e .
```

### Option 2: With All Dependencies
```bash
# Install with all optional dependencies
pip install -e ".[all]"
```

### Option 3: Selective Installation
```bash
# Core installation only
pip install -e .

# Add OpenAI support
pip install -e ".[openai]"

# Add SentenceTransformers support
pip install -e ".[sbert]"

# Add development tools
pip install -e ".[dev]"
```

## Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/tungnk99/rag-lib.git
cd rag-lib

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate  # or venv-dev\Scripts\activate on Windows

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

## Optional Dependencies

RAG-Lib uses optional dependencies to keep the core installation lightweight:

### Embedding Models
```bash
# OpenAI embeddings
pip install "rag-lib[openai]"

# Local SentenceTransformers
pip install "rag-lib[sbert]"
```

### Document Stores
```bash
# Qdrant vector database
pip install qdrant-client
```

### Development Tools
```bash
# Testing and code quality
pip install "rag-lib[dev]"
```

### Specialized Readers
```bash
# PDF processing (if you need advanced PDF features)
pip install pymupdf  # Alternative to PyPDF2

# Excel processing with more features
pip install xlsxwriter openpyxl

# Word document processing
pip install python-docx
```

## Verification

After installation, verify everything works:

### Basic Verification
```python
# Test basic imports
from rag_lib import __version__, OpenAIEmbedding, Query, Document
print(f"RAG-Lib version: {__version__}")

# Test schema creation
query = Query(content="Test query")
doc = Document(id="test", content="Test document")
print("✅ Basic functionality working!")
```

### Run Test Suite
```bash
# Run the test examples
python examples/test_rag_components.py

# If you installed dev dependencies
pytest tests/ -v
```

### Test Specific Components
```python
# Test document store
from rag_lib.document_stores import InMemoryDocumentStore
store = InMemoryDocumentStore()
store.create_index("test", vector_size=384)
print("✅ Document store working!")

# Test chunkers
from rag_lib.chunkers import create_base_length_chunker
chunker = create_base_length_chunker(max_tokens=512)
print("✅ Chunkers working!")

# Test readers
from rag_lib.readers import get_supported_formats
formats = get_supported_formats()
print(f"✅ Supported formats: {list(formats.keys())}")
```

## Environment Setup

### Environment Variables
Create a `.env` file in your project root:

```bash
# OpenAI API (if using OpenAI embeddings/LLMs)
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant (if using Qdrant vector store)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# Logging level
LOG_LEVEL=INFO

# Cache directory
RAG_CACHE_DIR=/tmp/rag_cache
```

### Configuration Files
Example configuration structure:

```
your_project/
├── configs/
│   ├── development.json
│   ├── production.json
│   └── testing.json
├── data/
│   ├── documents/
│   └── embeddings/
├── .env
└── main.py
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# If you get "ModuleNotFoundError"
import sys
sys.path.append('/path/to/rag-lib/src')
```

#### 2. OpenAI API Issues
```bash
# Verify API key
export OPENAI_API_KEY="your-key-here"
python -c "import openai; print(openai.api_key)"
```

#### 3. SentenceTransformers Issues
```bash
# If models don't download
export HF_HOME=/path/to/huggingface/cache
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### 4. Memory Issues
```python
# Reduce batch sizes in config
config = {
    "chunker": {"type": "length", "max_tokens": 256},
    "batch_size": 5  # Reduce this
}
```

#### 5. Permission Issues (Linux/macOS)
```bash
# Fix permissions
sudo chown -R $USER:$USER /path/to/rag-lib
chmod -R 755 /path/to/rag-lib
```

### Performance Optimization

#### For Large Documents
```python
# Use streaming and smaller chunks
config = {
    "chunker": {
        "type": "length",
        "max_tokens": 512,
        "overlap_tokens": 50
    },
    "batch_size": 10,
    "enable_progress": True
}
```

#### For Limited Memory
```python
# Use lighter embedding models
from rag_lib.models.embedding import SBERTEmbedding
embedding_model = SBERTEmbedding(
    model_name="all-MiniLM-L6-v2"  # Smaller model
)
```

### Getting Help

1. **Check Examples**: Look at files in the `examples/` directory
2. **GitHub Issues**: [Create an issue](https://github.com/tungnk99/rag-lib/issues)
3. **Documentation**: See `docs/` directory for more guides
4. **Community**: Join our discussions

### Uninstallation

```bash
# Uninstall the package
pip uninstall rag-lib

# Remove virtual environment
deactivate
rm -rf venv  # or venv-dev
```

## Next Steps

After installation:

1. **Read the [Getting Started Guide](docs/getting_started.md)**
2. **Try the [Quick Tutorial](docs/tutorials/basic_rag_pipeline.md)**
3. **Explore [Examples](examples/)**
4. **Check [API Reference](docs/api/)**

## Development Setup

For advanced users who want to contribute:

```bash
# Full development setup
git clone https://github.com/tungnk99/rag-lib.git
cd rag-lib

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate

# Install in editable mode with all dependencies
pip install -e ".[dev,all]"

# Install development tools
pip install pre-commit black isort flake8 mypy

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=rag_lib

# Run code quality checks
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

---

## Support

If you encounter any issues during installation, please:

1. Check this troubleshooting section
2. Look at existing [GitHub Issues](https://github.com/tungnk99/rag-lib/issues)
3. Create a new issue with:
   - Your operating system
   - Python version
   - Full error message
   - Steps to reproduce

We're here to help! 🚀
