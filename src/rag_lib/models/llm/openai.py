"""
OpenAI Large Language Model implementation.

This module provides an implementation of the LLMModel interface for OpenAI's GPT models.
It supports both completion and chat completion endpoints.
"""

from typing import List, Dict, Any, Optional, Union, Iterator
import warnings

from ._base import (
    LLMModel,
    Message,
    CompletionResponse,
    ChatCompletionResponse
)


class OpenAILLM(LLMModel):
    """
    OpenAI LLM implementation using OpenAI's API.
    
    This class provides access to OpenAI's GPT models including GPT-3.5 and GPT-4
    for both text completion and chat completion tasks.
    
    Attributes:
        api_key (str): OpenAI API key
        organization (Optional[str]): OpenAI organization ID
        base_url (Optional[str]): Custom base URL for API requests
        timeout (float): Request timeout in seconds
        max_retries (int): Maximum number of retries for failed requests
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        max_retries: int = 3,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            model_name (str): OpenAI model name (e.g., 'gpt-3.5-turbo', 'gpt-4')
            api_key (Optional[str]): OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            organization (Optional[str]): OpenAI organization ID
            base_url (Optional[str]): Custom base URL for API requests
            max_tokens (int): Maximum number of tokens to generate
            timeout (float): Request timeout in seconds
            max_retries (int): Maximum number of retries for failed requests
            config (Optional[Dict[str, Any]]): Additional configuration parameters
        """
        super().__init__(model_name=model_name, max_tokens=max_tokens, config=config)
        
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        self._init_client()
        
        # Validate model name
        self._validate_model_name()
    
    def _init_client(self):
        """Initialize the OpenAI client."""
        try:
            import openai
            
            client_kwargs = {}
            
            if self.api_key:
                client_kwargs['api_key'] = self.api_key
            
            if self.organization:
                client_kwargs['organization'] = self.organization
            
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
            
            if self.timeout:
                client_kwargs['timeout'] = self.timeout
            
            if self.max_retries:
                client_kwargs['max_retries'] = self.max_retries
            
            self.client = openai.OpenAI(**client_kwargs)
            
        except ImportError:
            raise ImportError(
                "OpenAI library is required for OpenAI LLM. "
                "Please install it with: pip install openai"
            )
    
    def _validate_model_name(self):
        """Validate that the model name is supported."""
        # List of supported OpenAI models (this could be extended)
        supported_models = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "text-davinci-003",
            "text-davinci-002",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001"
        ]
        
        # Allow any model name that starts with known prefixes (for newer models)
        model_prefixes = ["gpt-3.5", "gpt-4", "text-davinci", "text-curie", "text-babbage", "text-ada"]
        
        is_supported = (
            self.model_name in supported_models or
            any(self.model_name.startswith(prefix) for prefix in model_prefixes)
        )
        
        if not is_supported:
            warnings.warn(
                f"Model '{self.model_name}' may not be supported. "
                f"Supported models include: {', '.join(supported_models[:5])}..."
            )
    
    def validate_input_length(self, text: str) -> None:
        """
        Validate input text length for OpenAI model constraints.
        
        Args:
            text (str): Text to validate
            
        Raises:
            ValueError: If text exceeds model limits
        """
        # Rough token estimation (1 token ≈ 4 characters for English text)
        estimated_tokens = len(text) // 4
        
        # Model-specific token limits (approximate)
        token_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo-preview": 128000,
            "gpt-4-1106-preview": 128000,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
        }
        
        # Get limit for current model (default to 4096 if unknown)
        limit = token_limits.get(self.model_name, 4096)
        
        if estimated_tokens > limit:
            raise ValueError(
                f"Estimated tokens ({estimated_tokens}) exceed model limit ({limit}) "
                f"for model '{self.model_name}'"
            )
    
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
        Generate text completion using OpenAI's completion endpoint.
        
        Args:
            prompt (str): Input prompt text
            max_tokens (Optional[int]): Maximum tokens to generate
            temperature (float): Sampling temperature (0.0 to 2.0)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            stop (Optional[Union[str, List[str]]]): Stop sequences
            stream (bool): Whether to stream the response
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Union[CompletionResponse, Iterator[CompletionResponse]]: 
                Completion response or iterator for streaming
        """
        # Validate inputs
        self._validate_text_input(prompt)
        self.validate_input_length(prompt)
        self._validate_generation_params(max_tokens, temperature, top_p)
        
        # Use default max_tokens if not provided
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        try:
            # Prepare request parameters
            request_params = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
                **kwargs
            }
            
            if stop is not None:
                request_params["stop"] = stop
            
            # Make API call
            if stream:
                return self._handle_completion_stream(request_params)
            else:
                response = self.client.completions.create(**request_params)
                return self._parse_completion_response(response)
        
        except Exception as e:
            raise Exception(f"OpenAI completion failed: {str(e)}")
    
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
        Generate chat completion using OpenAI's chat completion endpoint.
        
        Args:
            messages (List[Message]): List of conversation messages
            max_tokens (Optional[int]): Maximum tokens to generate
            temperature (float): Sampling temperature (0.0 to 2.0)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            stop (Optional[Union[str, List[str]]]): Stop sequences
            stream (bool): Whether to stream the response
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Union[ChatCompletionResponse, Iterator[ChatCompletionResponse]]: 
                Chat completion response or iterator for streaming
        """
        # Validate inputs
        self._validate_messages_input(messages)
        self._validate_generation_params(max_tokens, temperature, top_p)
        
        # Validate total conversation length
        total_text = " ".join([msg.content for msg in messages])
        self.validate_input_length(total_text)
        
        # Use default max_tokens if not provided
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Prepare request parameters
            request_params = {
                "model": self.model_name,
                "messages": openai_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
                **kwargs
            }
            
            if stop is not None:
                request_params["stop"] = stop
            
            # Make API call
            if stream:
                return self._handle_chat_completion_stream(request_params)
            else:
                response = self.client.chat.completions.create(**request_params)
                return self._parse_chat_completion_response(response)
        
        except Exception as e:
            raise Exception(f"OpenAI chat completion failed: {str(e)}")
    
    def _parse_completion_response(self, response) -> CompletionResponse:
        """Parse OpenAI completion response."""
        choice = response.choices[0]
        
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }
        
        return CompletionResponse(
            text=choice.text,
            finish_reason=choice.finish_reason or "stop",
            usage=usage_info,
            model_name=self.model_name,
            metadata={"response_id": response.id, "created": response.created}
        )
    
    def _parse_chat_completion_response(self, response) -> ChatCompletionResponse:
        """Parse OpenAI chat completion response."""
        choice = response.choices[0]
        message = choice.message
        
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }
        
        response_message = Message(
            role=message.role,
            content=message.content
        )
        
        return ChatCompletionResponse(
            message=response_message,
            finish_reason=choice.finish_reason or "stop",
            usage=usage_info,
            model_name=self.model_name,
            metadata={"response_id": response.id, "created": response.created}
        )
    
    def _handle_completion_stream(self, request_params) -> Iterator[CompletionResponse]:
        """Handle streaming completion response."""
        try:
            stream = self.client.completions.create(**request_params)
            
            for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.text:
                        yield CompletionResponse(
                            text=choice.text,
                            finish_reason=choice.finish_reason or "continue",
                            model_name=self.model_name,
                            metadata={"chunk_id": chunk.id, "created": chunk.created}
                        )
        
        except Exception as e:
            raise Exception(f"OpenAI completion streaming failed: {str(e)}")
    
    def _handle_chat_completion_stream(self, request_params) -> Iterator[ChatCompletionResponse]:
        """Handle streaming chat completion response."""
        try:
            stream = self.client.chat.completions.create(**request_params)
            
            for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        response_message = Message(
                            role=choice.delta.role or "assistant",
                            content=choice.delta.content
                        )
                        
                        yield ChatCompletionResponse(
                            message=response_message,
                            finish_reason=choice.finish_reason or "continue",
                            model_name=self.model_name,
                            metadata={"chunk_id": chunk.id, "created": chunk.created}
                        )
        
        except Exception as e:
            raise Exception(f"OpenAI chat completion streaming failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information and configuration."""
        base_info = super().get_model_info()
        base_info.update({
            "api_provider": "openai",
            "organization": self.organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        })
        return base_info
