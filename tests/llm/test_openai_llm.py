"""
Test cases for OpenAI LLM implementation.

These tests demonstrate the usage and validate the functionality
of the OpenAI LLM class.
"""

import pytest
import os
from unittest.mock import Mock, patch

from src.rag_lib.models.llm import (
    OpenAILLM,
    Message,
    CompletionResponse,
    ChatCompletionResponse,
    create_user_message,
    create_system_message
)


class TestOpenAILLM:
    """Test cases for OpenAI LLM implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.model_name = "gpt-3.5-turbo"
        
    def test_initialization(self):
        """Test OpenAI LLM initialization."""
        llm = OpenAILLM(
            model_name=self.model_name,
            api_key=self.api_key,
            max_tokens=1000
        )
        
        assert llm.model_name == self.model_name
        assert llm.max_tokens == 1000
        assert llm.api_key == self.api_key
        
    def test_model_validation(self):
        """Test model name validation."""
        # Valid model should not raise warning
        llm = OpenAILLM(
            model_name="gpt-3.5-turbo",
            api_key=self.api_key
        )
        assert llm.model_name == "gpt-3.5-turbo"
        
    def test_input_length_validation(self):
        """Test input length validation."""
        llm = OpenAILLM(
            model_name="gpt-3.5-turbo",
            api_key=self.api_key
        )
        
        # Short text should pass
        short_text = "This is a short text."
        llm.validate_input_length(short_text)  # Should not raise
        
        # Very long text should raise ValueError
        very_long_text = "x" * 20000  # Much longer than token limit
        with pytest.raises(ValueError, match="exceed model limit"):
            llm.validate_input_length(very_long_text)
    
    @patch('src.rag_lib.models.llm.openai.openai.OpenAI')
    def test_completion_success(self, mock_openai):
        """Test successful completion."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].text = "This is a test completion."
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        
        mock_client.completions.create.return_value = mock_response
        
        # Create LLM and test completion
        llm = OpenAILLM(
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        result = llm.completion(
            prompt="Test prompt",
            temperature=0.7,
            max_tokens=50
        )
        
        assert isinstance(result, CompletionResponse)
        assert result.text == "This is a test completion."
        assert result.finish_reason == "stop"
        assert result.usage["total_tokens"] == 25
        
    @patch('src.rag_lib.models.llm.openai.openai.OpenAI')
    def test_chat_completion_success(self, mock_openai):
        """Test successful chat completion."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "This is a test chat response."
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 45
        mock_response.id = "test-chat-id"
        mock_response.created = 1234567890
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create LLM and test chat completion
        llm = OpenAILLM(
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        messages = [
            create_system_message("You are a helpful assistant."),
            create_user_message("Hello!")
        ]
        
        result = llm.chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=50
        )
        
        assert isinstance(result, ChatCompletionResponse)
        assert result.message.content == "This is a test chat response."
        assert result.message.role == "assistant"
        assert result.finish_reason == "stop"
        assert result.usage["total_tokens"] == 45
        
    def test_completion_input_validation(self):
        """Test completion input validation."""
        llm = OpenAILLM(
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        # Empty prompt should raise ValueError
        with pytest.raises(TypeError):
            llm.completion(prompt=None)
            
        with pytest.raises(ValueError):
            llm.completion(prompt="")
            
        # Invalid temperature should raise ValueError
        with pytest.raises(ValueError):
            llm.completion(prompt="test", temperature=3.0)
            
        # Invalid top_p should raise ValueError
        with pytest.raises(ValueError):
            llm.completion(prompt="test", top_p=1.5)
    
    def test_chat_completion_input_validation(self):
        """Test chat completion input validation."""
        llm = OpenAILLM(
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        # Empty messages should raise ValueError
        with pytest.raises(ValueError):
            llm.chat_completion(messages=[])
            
        # Invalid message type should raise TypeError
        with pytest.raises(TypeError):
            llm.chat_completion(messages=["not a message"])
    
    def test_get_model_info(self):
        """Test getting model information."""
        llm = OpenAILLM(
            model_name=self.model_name,
            api_key=self.api_key,
            organization="test-org",
            timeout=60.0,
            max_retries=5
        )
        
        info = llm.get_model_info()
        
        assert info["model_name"] == self.model_name
        assert info["api_provider"] == "openai"
        assert info["organization"] == "test-org"
        assert info["timeout"] == 60.0
        assert info["max_retries"] == 5


class TestUtilityFunctions:
    """Test utility functions for message creation."""
    
    def test_create_user_message(self):
        """Test creating user message."""
        message = create_user_message("Hello!")
        
        assert isinstance(message, Message)
        assert message.role == "user"
        assert message.content == "Hello!"
        
    def test_create_system_message(self):
        """Test creating system message."""
        message = create_system_message("You are helpful.")
        
        assert isinstance(message, Message)
        assert message.role == "system"
        assert message.content == "You are helpful."
        
    def test_message_validation(self):
        """Test message validation."""
        # Invalid role should raise ValueError
        with pytest.raises(ValueError):
            Message(role="invalid", content="test")
            
        # Empty content should raise ValueError
        with pytest.raises(ValueError):
            Message(role="user", content="")


if __name__ == "__main__":
    pytest.main([__file__])
