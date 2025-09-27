"""
OpenAI embedding model implementation.

This module provides OpenAI-based embedding models using their API.
Supports various OpenAI embedding models like text-embedding-ada-002, text-embedding-3-small, etc.
"""

import time
from typing import List, Optional, Dict, Any
import logging

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ._base import EmbeddingModel

logger = logging.getLogger(__name__)


class OpenAIEmbedding(EmbeddingModel):
    """
    OpenAI embedding model implementation.
    
    This class provides access to OpenAI's embedding models through their API.
    Supports rate limiting, retries, and proper error handling.
    
    Attributes:
        client (OpenAI): OpenAI API client
        model_name (str): OpenAI model name (e.g., 'text-embedding-ada-002')
        api_key (str): OpenAI API key
        max_tokens (int): Maximum tokens per request
        rate_limit_delay (float): Delay between requests to respect rate limits
    """
    
    # OpenAI model configurations
    MODEL_CONFIGS = {
        "text-embedding-ada-002": {
            "dimension": 1536,
            "max_tokens": 8192,
            "cost_per_1k_tokens": 0.0001
        },
        "text-embedding-3-small": {
            "dimension": 1536,
            "max_tokens": 8192,
            "cost_per_1k_tokens": 0.00002
        },
        "text-embedding-3-large": {
            "dimension": 3072,
            "max_tokens": 8192,
            "cost_per_1k_tokens": 0.00013
        }
    }
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit_delay: float = 0.1,
        max_retries: int = 3,
        timeout: float = 30.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model_name (str): OpenAI model name
            api_key (Optional[str]): OpenAI API key. If None, will use OPENAI_API_KEY env var
            organization (Optional[str]): OpenAI organization ID
            base_url (Optional[str]): Custom base URL for API
            rate_limit_delay (float): Delay between requests in seconds
            max_retries (int): Maximum number of retries for failed requests
            timeout (float): Request timeout in seconds
            config (Optional[Dict[str, Any]]): Additional configuration
            
        Raises:
            ImportError: If openai package is not installed
            ValueError: If model_name is not supported
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not found. Install it with: pip install openai"
            )
        
        if model_name not in self.MODEL_CONFIGS:
            available_models = list(self.MODEL_CONFIGS.keys())
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Available models: {available_models}"
            )
        
        model_config = self.MODEL_CONFIGS[model_name]
        
        # Initialize base class
        super().__init__(
            model_name=model_name,
            dimension=model_config["dimension"],
            config=config or {}
        )
        
        # OpenAI-specific attributes
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self.max_tokens = model_config["max_tokens"]
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize OpenAI client
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries
        }
        
        if organization:
            client_kwargs["organization"] = organization
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = OpenAI(**client_kwargs)
        
        logger.info(f"Initialized OpenAI embedding model: {model_name}")
    
    def validate_input_length(self, text: str) -> None:
        """
        Validate text length against OpenAI token limits.
        
        Args:
            text (str): Text to validate
            
        Raises:
            ValueError: If text exceeds token limits
            
        Note:
            This does a rough estimation based on character count.
            For exact token counting, you'd need tiktoken library.
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        estimated_tokens = len(text) // 4
        
        if estimated_tokens > self.max_tokens:
            raise ValueError(
                f"Estimated token count ({estimated_tokens}) exceeds "
                f"OpenAI model limit ({self.max_tokens}). "
                f"Consider chunking your text."
            )
    
    def encode_text(self, text: str, **kwargs) -> List[float]:
        """
        Encode a single text using OpenAI API.
        
        Args:
            text (str): Input text to encode
            **kwargs: Additional parameters for OpenAI API
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            ValueError: If text is invalid
            Exception: If API call fails
        """
        # Basic validation
        self._validate_text_input(text)
        self.validate_input_length(text)
        
        try:
            # Rate limiting
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            # Make API call
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                **kwargs
            )
            
            embedding = response.data[0].embedding
            
            logger.debug(f"Successfully encoded text of length {len(text)}")
            return embedding
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise Exception(f"Rate limit exceeded. Please try again later: {e}")
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            raise Exception(f"Failed to encode text: {e}")
    
    def batch_encode_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Encode multiple texts using OpenAI API.
        
        Args:
            texts (List[str]): List of input texts to encode
            **kwargs: Additional parameters for OpenAI API
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ValueError: If any text is invalid
            Exception: If API call fails
        """
        # Validate inputs
        self._validate_batch_input(texts)
        for text in texts:
            self.validate_input_length(text)
        
        try:
            # Rate limiting
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            # Make batch API call
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                **kwargs
            )
            
            # Extract embeddings in order
            embeddings = [item.embedding for item in response.data]
            
            logger.debug(f"Successfully encoded {len(texts)} texts")
            return embeddings
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise Exception(f"Rate limit exceeded. Please try again later: {e}")
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error during batch embedding: {e}")
            raise Exception(f"Failed to encode texts: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dict[str, Any]: Model information including OpenAI-specific details
        """
        base_info = super().get_model_info()
        
        openai_info = {
            "max_tokens": self.max_tokens,
            "rate_limit_delay": self.rate_limit_delay,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "cost_per_1k_tokens": self.MODEL_CONFIGS[self.model_name]["cost_per_1k_tokens"],
            "provider": "OpenAI"
        }
        
        return {**base_info, **openai_info}
    
    def estimate_cost(self, texts: List[str]) -> float:
        """
        Estimate the cost of encoding given texts.
        
        Args:
            texts (List[str]): List of texts to estimate cost for
            
        Returns:
            float: Estimated cost in USD
        """
        # Rough token estimation
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars // 4
        
        cost_per_1k = self.MODEL_CONFIGS[self.model_name]["cost_per_1k_tokens"]
        estimated_cost = (estimated_tokens / 1000) * cost_per_1k
        
        return estimated_cost
    
    def __repr__(self) -> str:
        """String representation of the OpenAI embedding model."""
        return f"OpenAIEmbedding(model='{self.model_name}', dimension={self.dimension})"
