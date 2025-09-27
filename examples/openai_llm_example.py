"""
Example usage of OpenAI LLM implementation.

This example demonstrates how to use the OpenAI LLM for both completion
and chat completion tasks.
"""

import os
from src.rag_lib.models.llm import (
    OpenAILLM,
    create_user_message,
    create_system_message,
    create_assistant_message
)


def main():
    """Main example function."""
    
    # Initialize OpenAI LLM
    # Make sure to set your OpenAI API key as environment variable: OPENAI_API_KEY
    llm = OpenAILLM(
        model_name="gpt-3.5-turbo",
        max_tokens=150,
        api_key=os.getenv("OPENAI_API_KEY")  # Or pass your API key directly
    )
    
    print("=== OpenAI LLM Example ===\n")
    
    # Example 1: Simple text completion
    print("1. Text Completion Example:")
    print("-" * 30)
    
    prompt = "The benefits of artificial intelligence include"
    
    try:
        completion_response = llm.completion(
            prompt=prompt,
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion_response.text}")
        print(f"Finish Reason: {completion_response.finish_reason}")
        print(f"Usage: {completion_response.usage}")
        print()
        
    except Exception as e:
        print(f"Error in completion: {e}")
        print()
    
    # Example 2: Chat completion
    print("2. Chat Completion Example:")
    print("-" * 30)
    
    # Create conversation messages
    messages = [
        create_system_message("You are a helpful AI assistant that provides clear and concise answers."),
        create_user_message("What is machine learning?"),
    ]
    
    try:
        chat_response = llm.chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        print("Conversation:")
        for msg in messages:
            print(f"{msg.role.capitalize()}: {msg.content}")
        
        print(f"Assistant: {chat_response.message.content}")
        print(f"Finish Reason: {chat_response.finish_reason}")
        print(f"Usage: {chat_response.usage}")
        print()
        
    except Exception as e:
        print(f"Error in chat completion: {e}")
        print()
    
    # Example 3: Multi-turn conversation
    print("3. Multi-turn Conversation Example:")
    print("-" * 35)
    
    conversation = [
        create_system_message("You are a helpful AI assistant."),
        create_user_message("What is Python?"),
    ]
    
    try:
        # First response
        response1 = llm.chat_completion(
            messages=conversation,
            temperature=0.7,
            max_tokens=80
        )
        
        # Add assistant's response to conversation
        conversation.append(response1.message)
        
        # Add follow-up question
        conversation.append(create_user_message("Can you give me a simple example?"))
        
        # Second response
        response2 = llm.chat_completion(
            messages=conversation,
            temperature=0.7,
            max_tokens=100
        )
        
        print("Full Conversation:")
        print(f"User: What is Python?")
        print(f"Assistant: {response1.message.content}")
        print(f"User: Can you give me a simple example?")
        print(f"Assistant: {response2.message.content}")
        print()
        
    except Exception as e:
        print(f"Error in multi-turn conversation: {e}")
        print()
    
    # Example 4: Streaming completion
    print("4. Streaming Completion Example:")
    print("-" * 32)
    
    try:
        stream_prompt = "Write a short poem about artificial intelligence:"
        
        print(f"Prompt: {stream_prompt}")
        print("Streaming response: ", end="", flush=True)
        
        for chunk in llm.completion(
            prompt=stream_prompt,
            temperature=0.8,
            max_tokens=80,
            stream=True
        ):
            print(chunk.text, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"Error in streaming: {e}")
        print()
    
    # Example 5: Model information
    print("5. Model Information:")
    print("-" * 20)
    
    model_info = llm.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
