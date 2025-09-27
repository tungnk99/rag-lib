"""
Large Language Model (LLM) implementations for the RAG system.

This package provides abstract base classes and concrete implementations
for various LLM providers and models.
"""

from ._base import (
    LLMModel,
    Message,
    CompletionResponse,
    ChatCompletionResponse,
    create_message,
    create_system_message,
    create_user_message,
    create_assistant_message
)

from .openai import OpenAILLM

__all__ = [
    # Base classes and data structures
    'LLMModel',
    'Message',
    'CompletionResponse',
    'ChatCompletionResponse',
    
    # Utility functions
    'create_message',
    'create_system_message',
    'create_user_message',
    'create_assistant_message',
    
    # LLM implementations
    'OpenAILLM',
]
