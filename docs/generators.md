# Generators

This module provides text generation capabilities for the RAG (Retrieval-Augmented Generation) system. Generators take queries and retrieved documents as input and produce contextual, coherent responses.

## Overview

The generators module consists of:
- **BaseGenerator**: Abstract base class defining the generator interface
- **GenerationConfig**: Configuration for generation parameters
- **GenerationResult**: Container for generation outputs and metadata
- **PromptTemplate**: Template system for building prompts
- **Pre-built templates**: Common prompt templates for different use cases

## Core Classes

### BaseGenerator

Abstract base class that all concrete generators must implement.

**Key Methods:**
- `generate(query, documents, config)`: Main generation method
- `generate_rag_response(query, retrieval_results, config)`: Generate complete RAG response
- `build_prompt(query, documents, template)`: Build formatted prompts
- `format_context(documents)`: Format documents into context string
- `validate_inputs(query, documents)`: Validate inputs before generation

**Example Implementation:**
```python
from rag_lib.generators import BaseGenerator, GenerationConfig, GenerationResult

class CustomGenerator(BaseGenerator):
    def generate(self, query, documents, config=None):
        # Validate inputs
        self.validate_inputs(query, documents)
        
        # Use config or default
        gen_config = config or self.config
        
        # Build prompt
        prompt = self.build_prompt(query, documents)
        
        # Your generation logic here
        generated_text = your_generation_function(prompt, gen_config)
        
        # Create result
        return GenerationResult(
            generated_text=generated_text,
            metadata=self.get_generation_metadata(query, documents, gen_config)
        )
    
    def is_available(self):
        return True  # Check if your generator is ready
```

### GenerationConfig

Configuration class for controlling generation parameters.

**Parameters:**
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (0.0-1.0)
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `stop_sequences`: Stop generation sequences
- `frequency_penalty`: Frequency penalty for repetition
- `presence_penalty`: Presence penalty for repetition
- `stream`: Whether to stream response

**Example:**
```python
from rag_lib.generators import GenerationConfig

# Basic configuration
config = GenerationConfig(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9
)

# Creative configuration
creative_config = GenerationConfig(
    max_tokens=1024,
    temperature=0.9,
    top_p=0.9,
    frequency_penalty=0.1
)

# Precise configuration
precise_config = GenerationConfig(
    max_tokens=256,
    temperature=0.1,
    top_p=0.8
)
```

### GenerationResult

Container for generation outputs and metadata.

**Attributes:**
- `generated_text`: The generated response text
- `metadata`: Generation metadata and statistics
- `finish_reason`: Why generation stopped
- `usage_info`: Token usage information
- `created_at`: Timestamp of generation

**Example:**
```python
result = generator.generate(query, documents)

print(f"Response: {result.generated_text}")
print(f"Tokens used: {result.usage_info.get('total_tokens', 'N/A')}")
print(f"Finish reason: {result.finish_reason}")
```

### PromptTemplate

Template system for building consistent prompts with placeholders.

**Features:**
- Variable substitution with `{variable}` syntax
- Required variables validation
- Automatic variable extraction from templates

**Example:**
```python
from rag_lib.generators import PromptTemplate

# Create custom template
template = PromptTemplate(
    template="""
    You are an expert in {domain}. Answer this question: {query}
    
    Context: {context}
    
    Answer: """,
    required_variables=['domain', 'query', 'context']
)

# Use template
prompt = template.format(
    domain="Machine Learning",
    query="What is supervised learning?",
    context="Supervised learning uses labeled data..."
)
```

## Pre-built Templates

### DEFAULT_QA_TEMPLATE
Standard question-answering template.

```python
from rag_lib.generators import DEFAULT_QA_TEMPLATE

prompt = generator.build_prompt(query, documents, DEFAULT_QA_TEMPLATE.template)
```

### SUMMARIZATION_TEMPLATE
Template for document summarization tasks.

```python
from rag_lib.generators import SUMMARIZATION_TEMPLATE

# Use for summarization
prompt = generator.build_prompt(query, documents, SUMMARIZATION_TEMPLATE.template)
```

### CONVERSATIONAL_TEMPLATE
Template for conversational responses.

```python
from rag_lib.generators import CONVERSATIONAL_TEMPLATE

# Use for chat-like interactions
prompt = generator.build_prompt(query, documents, CONVERSATIONAL_TEMPLATE.template)
```

## Pre-configured Settings

Common generation configurations are provided:

```python
from rag_lib.generators import DEFAULT_CONFIG, CREATIVE_CONFIG, PRECISE_CONFIG, SUMMARIZATION_CONFIG

# Use pre-configured settings
result = generator.generate(query, documents, CREATIVE_CONFIG)
```

## Usage Examples

### Basic Generation
```python
from rag_lib.generators import BaseGenerator, GenerationConfig
from rag_lib.schemas import create_query, create_document

# Create query and documents
query = create_query("What is machine learning?")
documents = [
    create_document("doc1", "Machine learning is...", {"title": "ML Intro"}),
    create_document("doc2", "ML algorithms include...", {"title": "ML Algorithms"})
]

# Initialize generator (implement your own or use existing)
generator = YourGenerator()

# Generate response
result = generator.generate(query, documents)
print(result.generated_text)
```

### Complete RAG Pipeline
```python
from rag_lib.generators import BaseGenerator
from rag_lib.schemas import RetrievalResult

# Assume you have retrieval results
retrieval_results = [
    RetrievalResult(document=doc1, score=0.9, rank=1),
    RetrievalResult(document=doc2, score=0.8, rank=2)
]

# Generate complete RAG response
rag_response = generator.generate_rag_response(
    query=query,
    retrieval_results=retrieval_results,
    config=PRECISE_CONFIG
)

print(f"Question: {rag_response.query.content}")
print(f"Answer: {rag_response.generated_text}")
print(f"Sources: {len(rag_response.retrieved_documents)} documents")
print(f"Generation time: {rag_response.generation_time:.3f}s")
```

### Custom Prompt Templates
```python
# Create domain-specific template
medical_template = PromptTemplate("""
You are a medical AI assistant. Based on the medical literature provided, 
answer the following question: {query}

Medical References:
{context}

Important: Always recommend consulting with healthcare professionals.

Medical Response: """, required_variables=['query', 'context'])

# Use custom template
prompt = generator.build_prompt(query, medical_docs, medical_template.template)
```

### Configuration Management
```python
# Dynamic configuration based on query type
def get_config_for_query(query):
    if "creative" in query.content.lower():
        return CREATIVE_CONFIG
    elif "summary" in query.content.lower():
        return SUMMARIZATION_CONFIG
    else:
        return PRECISE_CONFIG

# Use dynamic configuration
config = get_config_for_query(query)
result = generator.generate(query, documents, config)
```

### Error Handling
```python
try:
    # Validate before generation
    generator.validate_inputs(query, documents)
    
    # Check if generator is available
    if not generator.is_available():
        raise RuntimeError("Generator not available")
    
    # Generate response
    result = generator.generate(query, documents)
    
except ValueError as e:
    print(f"Input validation error: {e}")
except RuntimeError as e:
    print(f"Generator error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Integration with RAG Pipeline

Generators integrate seamlessly with other RAG components:

```python
# Complete RAG pipeline
def rag_pipeline(user_query, document_store, retriever, generator):
    # 1. Create query
    query = create_query(user_query)
    
    # 2. Retrieve documents
    retrieval_results = retriever.retrieve(query, document_store)
    
    # 3. Generate response
    rag_response = generator.generate_rag_response(query, retrieval_results)
    
    return rag_response

# Use pipeline
response = rag_pipeline(
    "What is machine learning?",
    my_document_store,
    my_retriever,
    my_generator
)
```

## Best Practices

### 1. Input Validation
Always validate inputs before generation:
```python
def generate(self, query, documents, config=None):
    self.validate_inputs(query, documents)
    # ... rest of generation logic
```

### 2. Configuration Management
Use appropriate configurations for different tasks:
```python
# For factual Q&A
factual_config = GenerationConfig(temperature=0.1, max_tokens=256)

# For creative tasks
creative_config = GenerationConfig(temperature=0.8, max_tokens=1024)
```

### 3. Context Formatting
Format context consistently:
```python
def format_context(self, documents):
    context_parts = []
    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get('title', f'Document {doc.id}')
        context_parts.append(f"[Source {i}: {title}]\n{doc.content}")
    return "\n\n".join(context_parts)
```

### 4. Metadata Collection
Collect comprehensive metadata:
```python
metadata = {
    'generator_class': self.__class__.__name__,
    'config': config.to_dict(),
    'document_count': len(documents),
    'context_length': len(context),
    'timestamp': datetime.now().isoformat()
}
```

### 5. Error Handling
Implement robust error handling:
```python
try:
    result = self.call_llm_api(prompt, config)
except APIError as e:
    return GenerationResult(
        generated_text="Sorry, I encountered an error generating a response.",
        metadata={'error': str(e)},
        finish_reason='error'
    )
```

## Extending the Generator System

To create a new generator, inherit from BaseGenerator and implement the required methods:

```python
class MyCustomGenerator(BaseGenerator):
    def __init__(self, api_key, model_name="gpt-3.5-turbo", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model_name = model_name
    
    def generate(self, query, documents, config=None):
        # Your implementation here
        pass
    
    def is_available(self):
        # Check if API key is valid, model is accessible, etc.
        return bool(self.api_key)
```

This generator system provides a flexible foundation for implementing various text generation approaches in RAG systems while maintaining consistency and extensibility.
