"""
Base class for Large Language Models (LLM).

This module provides the abstract base class for all LLM models in the RAG system.
LLM models are responsible for text generation, chat completion, and text completion.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ...schemas.schema import Query, Document


@dataclass
class Message:
    """
    Represents a message in a chat conversation.
    
    Attributes:
        role (str): Role of the message sender ('system', 'user', 'assistant')
        content (str): Content of the message
        metadata (Dict[str, Any]): Additional metadata for the message
    """
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate message after initialization."""
        if self.role not in ['system', 'user', 'assistant']:
            raise ValueError("Role must be one of: 'system', 'user', 'assistant'")
        
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create Message from dictionary representation."""
        return cls(**data)


@dataclass
class CompletionResponse:
    """
    Represents a completion response from an LLM.
    
    Attributes:
        text (str): Generated text
        finish_reason (str): Reason why generation finished ('stop', 'length', 'error')
        usage (Dict[str, Any]): Token usage information
        model_name (str): Name of the model used
        metadata (Dict[str, Any]): Additional response metadata
        response_id (str): Unique identifier for this response
        created_at (str): ISO timestamp when response was created
    """
    text: str
    finish_reason: str
    usage: Dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate completion response after initialization."""
        if self.finish_reason not in ['stop', 'length', 'error']:
            raise ValueError("finish_reason must be one of: 'stop', 'length', 'error'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert completion response to dictionary representation."""
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "model_name": self.model_name,
            "metadata": self.metadata,
            "response_id": self.response_id,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompletionResponse':
        """Create CompletionResponse from dictionary representation."""
        return cls(**data)


@dataclass
class ChatCompletionResponse:
    """
    Represents a chat completion response from an LLM.
    
    Attributes:
        message (Message): Generated message
        finish_reason (str): Reason why generation finished ('stop', 'length', 'error')
        usage (Dict[str, Any]): Token usage information
        model_name (str): Name of the model used
        metadata (Dict[str, Any]): Additional response metadata
        response_id (str): Unique identifier for this response
        created_at (str): ISO timestamp when response was created
    """
    message: Message
    finish_reason: str
    usage: Dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate chat completion response after initialization."""
        if self.finish_reason not in ['stop', 'length', 'error']:
            raise ValueError("finish_reason must be one of: 'stop', 'length', 'error'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chat completion response to dictionary representation."""
        return {
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "model_name": self.model_name,
            "metadata": self.metadata,
            "response_id": self.response_id,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatCompletionResponse':
        """Create ChatCompletionResponse from dictionary representation."""
        data = data.copy()
        data["message"] = Message.from_dict(data["message"])
        return cls(**data)


class LLMModel(ABC):
    """
    Abstract base class for Large Language Models.
    
    This class defines the interface for all LLM models, whether they are
    API-based (like OpenAI, Anthropic) or local models (like Hugging Face Transformers).
    
    Attributes:
        model_name (str): Name/identifier of the LLM model
        max_tokens (int): Maximum number of tokens the model can generate
        config (Dict[str, Any]): Model-specific configuration parameters
    """
    
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 2048,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LLM model.
        
        Args:
            model_name (str): Name/identifier of the LLM model
            max_tokens (int): Maximum number of tokens the model can generate
            config (Optional[Dict[str, Any]]): Model-specific configuration parameters
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.config = config or {}
    
    @abstractmethod
    def completion(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, Iterator[CompletionResponse]]:
        """
        Generate text completion for a given prompt.
        
        Args:
            prompt (str): Input prompt text
            max_tokens (Optional[int]): Maximum tokens to generate (overrides default)
            temperature (float): Sampling temperature (0.0 to 2.0)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            stop (Optional[Union[str, List[str]]]): Stop sequences
            stream (bool): Whether to stream the response
            **kwargs: Additional model-specific parameters
            
        Returns:
            Union[CompletionResponse, Iterator[CompletionResponse]]: 
                Completion response or iterator for streaming
            
        Raises:
            ValueError: If prompt is empty or parameters are invalid
            Exception: If completion generation fails
        """
        pass
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionResponse]]:
        """
        Generate chat completion for a conversation.
        
        Args:
            messages (List[Message]): List of conversation messages
            max_tokens (Optional[int]): Maximum tokens to generate (overrides default)
            temperature (float): Sampling temperature (0.0 to 2.0)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            stop (Optional[Union[str, List[str]]]): Stop sequences
            stream (bool): Whether to stream the response
            **kwargs: Additional model-specific parameters
            
        Returns:
            Union[ChatCompletionResponse, Iterator[ChatCompletionResponse]]: 
                Chat completion response or iterator for streaming
            
        Raises:
            ValueError: If messages are empty or parameters are invalid
            Exception: If chat completion generation fails
        """
        pass
    
    def validate_input_length(self, text: str) -> None:
        """
        Validate input text length for model-specific constraints.
        
        Args:
            text (str): Text to validate
            
        Raises:
            ValueError: If text exceeds model limits
            
        Note:
            This is a hook for concrete implementations to add model-specific
            validation like token limits. Default implementation does nothing.
            Override this method in concrete classes as needed.
        """
        # Default implementation does nothing
        # Concrete classes should override this for model-specific validation
        pass
    
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Dict[str, Any]: Model information including name, max_tokens, config, etc.
        """
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "config": self.config
        }
    
    def _validate_text_input(self, text: str) -> None:
        """
        Validate basic text input before generation.
        
        Args:
            text (str): Text to validate
            
        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
    
    def _validate_messages_input(self, messages: List[Message]) -> None:
        """
        Validate messages input for chat completion.
        
        Args:
            messages (List[Message]): Messages to validate
            
        Raises:
            ValueError: If messages are invalid
        """
        if not isinstance(messages, list):
            raise TypeError("Messages must be a list")
        
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for i, message in enumerate(messages):
            if not isinstance(message, Message):
                raise TypeError(f"Message at index {i} must be a Message object")
    
    def _validate_generation_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0
    ) -> None:
        """
        Validate generation parameters.
        
        Args:
            max_tokens (Optional[int]): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            
        Raises:
            ValueError: If parameters are invalid
        """
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
    
    
    def __repr__(self) -> str:
        """String representation of the LLM model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', max_tokens={self.max_tokens})"


# Utility functions for creating messages and responses

def create_message(role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
    """
    Convenience function to create a Message object.
    
    Args:
        role (str): Message role ('system', 'user', 'assistant')
        content (str): Message content
        metadata (Optional[Dict[str, Any]]): Message metadata
    
    Returns:
        Message: Created message object
    """
    return Message(role=role, content=content, metadata=metadata or {})


def create_system_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a system message."""
    return create_message("system", content, metadata)


def create_user_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a user message."""
    return create_message("user", content, metadata)


def create_assistant_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create an assistant message."""
    return create_message("assistant", content, metadata)
