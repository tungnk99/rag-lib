"""
Generators module for the RAG library.

This module provides text generation functionality for creating responses
in RAG systems. Generators take queries and retrieved documents to produce
coherent, contextual responses.

The module includes:
- BaseGenerator: Abstract base class for all generators
- GenerationConfig: Configuration for text generation parameters
- GenerationResult: Result of text generation operations
- PromptTemplate: Template system for building prompts
"""

from ._base import (
    BaseGenerator,
    GenerationConfig,
    GenerationResult
)
from .prompts import (
    PromptTemplate,
    DEFAULT_QA_TEMPLATE,
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_SUMMARY_TEMPLATE
)

__all__ = [
    "BaseGenerator",
    "GenerationConfig", 
    "GenerationResult",
    "PromptTemplate",
    "DEFAULT_QA_TEMPLATE",
    "DEFAULT_CHAT_TEMPLATE", 
    "DEFAULT_SUMMARY_TEMPLATE"
]
