# Configuration Guide

RAG-Lib supports flexible configuration through JSON, YAML, and Python dictionaries. This guide covers all configuration options and best practices.

## Table of Contents
- [Configuration Overview](#configuration-overview)
- [Pipeline Configuration](#pipeline-configuration)
- [Component Configuration](#component-configuration)
- [Environment-Specific Configs](#environment-specific-configs)
- [Best Practices](#best-practices)

## Configuration Overview

### Configuration Methods

1. **JSON Files**: Easy to read and edit
2. **YAML Files**: More human-readable, supports comments
3. **Python Dictionaries**: Programmatic configuration
4. **Environment Variables**: Sensitive data and deployment settings

### Basic Structure

```json
{
  "embedding_model": { "type": "openai", "model_name": "text-embedding-3-small" },
  "document_store": { "type": "inmemory" },
  "chunker": { "type": "recursive", "max_tokens": 1024 },
  "retriever": { "type": "semantic", "top_k": 10 },
  "ranker": { "type": "cross_encoder", "top_k": 5 },
  "generator": { "type": "openai", "model_name": "gpt-3.5-turbo" }
}
```

## Pipeline Configuration

### Data Pipeline Configuration

```json
{
  "chunker": {
    "type": "recursive",
    "max_tokens": 1024,
    "overlap_tokens": 100,
    "create_parent_chunks": true
  },
  "enable_progress": true,
  "batch_size": 10,
  "logger_config": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

### RAG Pipeline Configuration

```json
{
  "embedding_model": {
    "type": "openai",
    "model_name": "text-embedding-3-small",
    "api_key_env": "OPENAI_API_KEY",
    "batch_size": 100,
    "rate_limit_delay": 0.1
  },
  "document_store": {
    "type": "qdrant",
    "url": "http://localhost:6333",
    "api_key_env": "QDRANT_API_KEY",
    "collection_config": {
      "size": 1536,
      "distance": "Cosine"
    }
  },
  "retriever": {
    "type": "semantic",
    "top_k": 20,
    "similarity_threshold": 0.6,
    "enable_filters": true
  },
  "ranker": {
    "type": "cross_encoder",
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_k": 5,
    "batch_size": 32
  },
  "generator": {
    "type": "openai",
    "model_name": "gpt-3.5-turbo",
    "max_tokens": 500,
    "temperature": 0.1,
    "api_key_env": "OPENAI_API_KEY"
  }
}
```

## Component Configuration

### Embedding Models

#### OpenAI Embeddings
```json
{
  "embedding_model": {
    "type": "openai",
    "model_name": "text-embedding-3-small",
    "api_key_env": "OPENAI_API_KEY",
    "api_base": "https://api.openai.com/v1",
    "batch_size": 100,
    "rate_limit_delay": 0.1,
    "max_retries": 3,
    "timeout": 30
  }
}
```

#### SentenceTransformers Embeddings
```json
{
  "embedding_model": {
    "type": "sentence_transformer",
    "model_name": "all-mpnet-base-v2",
    "device": "cuda",
    "normalize_embeddings": true,
    "batch_size": 32,
    "cache_folder": "./models"
  }
}
```

### Document Stores

#### In-Memory Store
```json
{
  "document_store": {
    "type": "inmemory",
    "name": "default_store",
    "config": {
      "similarity_metric": "cosine",
      "enable_metadata_filters": true
    }
  }
}
```

#### Qdrant Store
```json
{
  "document_store": {
    "type": "qdrant",
    "url": "http://localhost:6333",
    "api_key_env": "QDRANT_API_KEY",
    "collection_config": {
      "size": 1536,
      "distance": "Cosine",
      "hnsw_config": {
        "m": 16,
        "ef_construct": 100
      }
    },
    "timeout": 30,
    "prefer_grpc": false
  }
}
```

### Chunkers

#### Length-Based Chunker
```json
{
  "chunker": {
    "type": "length",
    "max_tokens": 512,
    "overlap_tokens": 50,
    "preserve_sentences": true,
    "tokenizer": "gpt2"
  }
}
```

#### Recursive Chunker
```json
{
  "chunker": {
    "type": "recursive",
    "max_tokens": 1024,
    "overlap_tokens": 100,
    "create_parent_chunks": true,
    "levels": [
      {
        "separators": ["\n\n\n", "\n\n"],
        "chunk_size": 2048,
        "chunk_overlap": 200
      },
      {
        "separators": ["\n"],
        "chunk_size": 1024,
        "chunk_overlap": 100
      }
    ]
  }
}
```

#### Semantic Chunker
```json
{
  "chunker": {
    "type": "semantic_sentence_transformer",
    "model_name": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.7,
    "max_tokens": 512,
    "min_tokens": 50,
    "buffer_size": 1,
    "device": "cpu"
  }
}
```

### Retrievers

#### Semantic Retriever
```json
{
  "retriever": {
    "type": "semantic",
    "name": "main_retriever",
    "top_k": 10,
    "similarity_threshold": 0.6,
    "enable_filters": true,
    "filter_strategy": "pre_filter",
    "query_preprocessing": {
      "lowercase": true,
      "remove_punctuation": false,
      "expand_acronyms": false
    }
  }
}
```

### Rankers

#### Cross-Encoder Ranker
```json
{
  "ranker": {
    "type": "cross_encoder",
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_k": 5,
    "batch_size": 32,
    "device": "cuda",
    "score_threshold": 0.0
  }
}
```

#### NoOp Ranker
```json
{
  "ranker": {
    "type": "noop",
    "preserve_order": true,
    "add_rank_scores": false
  }
}
```

### Generators

#### OpenAI Generator
```json
{
  "generator": {
    "type": "openai",
    "model_name": "gpt-3.5-turbo",
    "api_key_env": "OPENAI_API_KEY",
    "max_tokens": 500,
    "temperature": 0.1,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop_sequences": ["###", "---"],
    "system_prompt": "You are a helpful assistant that answers questions based on provided context."
  }
}
```

## Environment-Specific Configs

### Development Configuration
```json
{
  "environment": "development",
  "debug": true,
  "embedding_model": {
    "type": "sentence_transformer",
    "model_name": "all-MiniLM-L6-v2"
  },
  "document_store": {
    "type": "inmemory"
  },
  "chunker": {
    "type": "length",
    "max_tokens": 256,
    "overlap_tokens": 25
  },
  "retriever": {
    "top_k": 5
  },
  "ranker": {
    "type": "noop"
  },
  "logging": {
    "level": "DEBUG",
    "handlers": ["console"],
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

### Production Configuration
```json
{
  "environment": "production",
  "debug": false,
  "embedding_model": {
    "type": "openai",
    "model_name": "text-embedding-3-small",
    "batch_size": 100,
    "rate_limit_delay": 0.1
  },
  "document_store": {
    "type": "qdrant",
    "url": "${QDRANT_URL}",
    "api_key_env": "QDRANT_API_KEY",
    "collection_config": {
      "size": 1536,
      "distance": "Cosine"
    }
  },
  "chunker": {
    "type": "recursive",
    "max_tokens": 1024,
    "overlap_tokens": 100
  },
  "retriever": {
    "top_k": 20,
    "similarity_threshold": 0.6
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
  },
  "logging": {
    "level": "WARNING",
    "handlers": ["file", "syslog"],
    "file_path": "/var/log/rag-lib.log",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
  },
  "monitoring": {
    "enable_metrics": true,
    "metrics_endpoint": "http://prometheus:9090",
    "alert_thresholds": {
      "query_time_p95": 5.0,
      "error_rate": 0.05
    }
  }
}
```

### Testing Configuration
```json
{
  "environment": "testing",
  "debug": true,
  "embedding_model": {
    "type": "mock",
    "dimension": 384
  },
  "document_store": {
    "type": "inmemory",
    "name": "test_store"
  },
  "chunker": {
    "type": "length",
    "max_tokens": 100,
    "overlap_tokens": 10
  },
  "retriever": {
    "top_k": 3
  },
  "ranker": {
    "type": "noop"
  },
  "generator": {
    "type": "mock",
    "response_template": "Test response based on: {context}"
  }
}
```

## Configuration Loading

### From JSON File
```python
import json
from rag_lib.pipelines.data_pipeline import create_pipeline_from_config

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Create pipeline
pipeline = create_pipeline_from_config(config, document_store)
```

### From YAML File
```python
import yaml
from rag_lib.pipelines import create_pipeline_from_config

# Load YAML configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

pipeline = create_pipeline_from_config(config, document_store)
```

### Environment Variable Substitution
```python
import os
import json

def load_config_with_env_vars(config_path: str) -> dict:
    """Load config with environment variable substitution."""
    
    with open(config_path, 'r') as f:
        config_str = f.read()
    
    # Replace environment variables
    for env_var in os.environ:
        config_str = config_str.replace(f"${{{env_var}}}", os.environ[env_var])
    
    return json.loads(config_str)

# Usage
config = load_config_with_env_vars('config.json')
```

### Configuration Validation
```python
from typing import Dict, Any

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure."""
    
    required_fields = ['embedding_model', 'document_store', 'chunker']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate embedding model
    if 'type' not in config['embedding_model']:
        raise ValueError("Embedding model must specify 'type'")
    
    # Validate chunker
    chunker_config = config['chunker']
    if chunker_config.get('max_tokens', 0) <= 0:
        raise ValueError("Chunker max_tokens must be positive")
    
    return True

# Usage
try:
    validate_config(config)
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Configuration Templates

### Small Documents Template
```json
{
  "name": "small_documents",
  "description": "Optimized for small documents (< 1000 words)",
  "chunker": {
    "type": "length",
    "max_tokens": 256,
    "overlap_tokens": 25,
    "preserve_sentences": true
  },
  "retriever": {
    "top_k": 15
  },
  "ranker": {
    "type": "cross_encoder",
    "top_k": 5
  },
  "batch_size": 20
}
```

### Large Documents Template
```json
{
  "name": "large_documents", 
  "description": "Optimized for large documents (> 10,000 words)",
  "chunker": {
    "type": "recursive",
    "max_tokens": 2048,
    "overlap_tokens": 200,
    "create_parent_chunks": true
  },
  "retriever": {
    "top_k": 30
  },
  "ranker": {
    "type": "cross_encoder",
    "top_k": 10
  },
  "batch_size": 5
}
```

### Code Documents Template
```json
{
  "name": "code_documents",
  "description": "Optimized for source code and technical documentation",
  "chunker": {
    "type": "recursive_code",
    "overlap_tokens": 25,
    "create_parent_chunks": true,
    "language_specific": true
  },
  "retriever": {
    "top_k": 10,
    "query_preprocessing": {
      "preserve_code_syntax": true,
      "expand_acronyms": false
    }
  },
  "ranker": {
    "type": "cross_encoder",
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_k": 5
  }
}
```

### Academic Papers Template
```json
{
  "name": "academic_papers",
  "description": "Optimized for academic and research papers",
  "chunker": {
    "type": "recursive_academic",
    "overlap_tokens": 75,
    "create_parent_chunks": true,
    "preserve_citations": true
  },
  "retriever": {
    "top_k": 20,
    "similarity_threshold": 0.7
  },
  "ranker": {
    "type": "cross_encoder",
    "top_k": 8
  },
  "generator": {
    "system_prompt": "You are an academic research assistant. Provide detailed, well-cited responses based on the provided research papers."
  }
}
```

## Dynamic Configuration

### Runtime Configuration Updates
```python
class ConfigurablePipeline:
    """Pipeline with runtime configuration updates."""
    
    def __init__(self, initial_config: dict):
        self.config = initial_config
        self.pipeline = self._create_pipeline()
    
    def update_config(self, updates: dict) -> None:
        """Update configuration at runtime."""
        
        # Deep merge configuration
        self._deep_merge(self.config, updates)
        
        # Recreate affected components
        if 'chunker' in updates:
            self._update_chunker()
        
        if 'retriever' in updates:
            self._update_retriever()
    
    def _deep_merge(self, base: dict, updates: dict) -> None:
        """Deep merge configuration dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
```

### A/B Testing Configuration
```python
import random
from typing import Dict, Any

class ABTestConfig:
    """Configuration for A/B testing different setups."""
    
    def __init__(self, variant_configs: Dict[str, Dict[str, Any]]):
        self.variants = variant_configs
        self.current_variant = None
    
    def get_config(self, user_id: str = None) -> Dict[str, Any]:
        """Get configuration for user (consistent assignment)."""
        
        if user_id:
            # Consistent assignment based on user ID
            hash_val = hash(user_id) % len(self.variants)
            variant_name = list(self.variants.keys())[hash_val]
        else:
            # Random assignment
            variant_name = random.choice(list(self.variants.keys()))
        
        self.current_variant = variant_name
        return self.variants[variant_name]

# Usage
ab_test = ABTestConfig({
    "variant_a": {
        "chunker": {"type": "length", "max_tokens": 512},
        "retriever": {"top_k": 10}
    },
    "variant_b": {
        "chunker": {"type": "recursive", "max_tokens": 1024},
        "retriever": {"top_k": 15}
    }
})

config = ab_test.get_config(user_id="user123")
```

## Best Practices

### 1. Environment Separation
```bash
# Use different config files for each environment
configs/
├── development.json
├── testing.json
├── staging.json
└── production.json
```

### 2. Sensitive Data Management
```python
# Never commit API keys to configuration files
# Use environment variables instead
{
  "embedding_model": {
    "type": "openai",
    "api_key_env": "OPENAI_API_KEY"  # Reference to env var
  }
}
```

### 3. Configuration Versioning
```json
{
  "config_version": "1.2.0",
  "created_at": "2024-01-15T10:30:00Z",
  "created_by": "data_team",
  "description": "Updated chunking strategy for better performance",
  "components": {
    // ... component configurations
  }
}
```

### 4. Configuration Documentation
```json
{
  "chunker": {
    "type": "recursive",
    "max_tokens": 1024,
    "_comment": "Increased from 512 to handle longer technical documents",
    "overlap_tokens": 100,
    "_rationale": "10% overlap provides good context without too much redundancy"
  }
}
```

### 5. Performance Tuning Guidelines

#### Memory-Constrained Environments
```json
{
  "embedding_model": {
    "type": "sentence_transformer",
    "model_name": "all-MiniLM-L6-v2",  // Smaller model
    "batch_size": 16  // Smaller batches
  },
  "chunker": {
    "max_tokens": 512,  // Smaller chunks
    "overlap_tokens": 25
  },
  "batch_size": 5  // Process fewer documents at once
}
```

#### High-Performance Environments
```json
{
  "embedding_model": {
    "type": "sentence_transformer", 
    "model_name": "all-mpnet-base-v2",  // Better quality
    "device": "cuda",
    "batch_size": 64  // Larger batches
  },
  "chunker": {
    "max_tokens": 2048,  // Larger chunks
    "overlap_tokens": 200
  },
  "batch_size": 50,  // Process more documents
  "parallel_processing": true
}
```

### 6. Monitoring Configuration
```json
{
  "monitoring": {
    "enable_metrics": true,
    "log_queries": true,
    "log_responses": false,  // Privacy consideration
    "performance_logging": {
      "log_slow_queries": true,
      "slow_query_threshold": 5.0,
      "sample_rate": 0.1
    },
    "error_tracking": {
      "capture_errors": true,
      "capture_context": true,
      "ignore_patterns": ["ConnectionError"]
    }
  }
}
```

This configuration guide should help you set up RAG-Lib for any environment or use case. Start with the templates and adjust based on your specific requirements.
