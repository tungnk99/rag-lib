"""
Base length-based chunker implementation.

This module provides a token-based chunker that splits text into chunks
based on maximum token count with optional overlap between chunks.
"""

from typing import List, Optional, Dict, Any, Callable
import re
from ._base import BaseChunker, ChunkerError
from ..schemas.schema import Document, FileContent, Element, ElementType


class TokenCountError(ChunkerError):
    """Raised when token counting fails."""
    pass


class BaseLengthChunker(BaseChunker):
    """
    Token-based length chunker that splits FileContent into Document chunks.
    
    This chunker splits text content based on token count limits with optional
    overlap between chunks to maintain context continuity.
    
    Attributes:
        max_tokens (int): Maximum number of tokens per chunk
        overlap_tokens (int): Number of tokens to overlap between chunks
        tokenizer (Callable): Function to tokenize text into tokens
        preserve_sentences (bool): Whether to try to preserve sentence boundaries
    """
    
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        preserve_sentences: bool = True,
        **kwargs
    ):
        """
        Initialize the length-based chunker.
        
        Args:
            max_tokens (int): Maximum number of tokens per chunk
            overlap_tokens (int): Number of tokens to overlap between chunks
            tokenizer (Optional[Callable]): Custom tokenizer function. If None, uses simple whitespace tokenizer
            preserve_sentences (bool): Whether to try to preserve sentence boundaries
            **kwargs: Additional configuration parameters
        """
        super().__init__(name="base_length_chunker", config=kwargs)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.preserve_sentences = preserve_sentences
        
        # Set up tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self._default_tokenizer
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate chunker configuration parameters."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """
        Default whitespace-based tokenizer.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Simple whitespace tokenizer with basic punctuation handling
        tokens = []
        for token in text.split():
            # Remove leading/trailing punctuation but keep it as separate tokens
            token = token.strip()
            if token:
                tokens.append(token)
        return tokens
    
    def _advanced_tokenizer(self, text: str) -> List[str]:
        """
        More advanced tokenizer that handles punctuation better.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Split on whitespace and punctuation
        pattern = r'\w+|[^\w\s]'
        tokens = re.findall(pattern, text)
        return [token for token in tokens if token.strip()]
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the configured tokenizer.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        try:
            tokens = self.tokenizer(text)
            return len(tokens)
        except Exception as e:
            raise TokenCountError(f"Failed to count tokens: {str(e)}")
    
    def chunk_file_content(self, file_content: FileContent) -> List[Document]:
        """
        Split file content into document chunks based on token count.
        
        Args:
            file_content (FileContent): File content to be chunked
            
        Returns:
            List[Document]: List of document chunks
            
        Raises:
            ChunkerError: If chunking fails
            ValueError: If file content is invalid
        """
        if not file_content.elements:
            raise ValueError("FileContent has no elements to chunk")
        
        # Extract all text content
        text_content = file_content.get_text_content()
        if not text_content.strip():
            raise ValueError("FileContent has no text content to chunk")
        
        # Get text elements for metadata
        text_elements = file_content.get_elements_by_type(ElementType.TEXT)
        element_metadata = self._merge_element_metadata(text_elements)
        
        # Split into chunks
        text_chunks = self._split_text_into_chunks(text_content)
        
        # Create document chunks
        documents = []
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                doc_chunk = self._create_document_chunk(
                    content=chunk_text,
                    file_content=file_content,
                    chunk_index=i,
                    element_metadata=element_metadata
                )
                # Add chunking-specific metadata
                doc_chunk.metadata.update({
                    "chunk_tokens": self.count_tokens(chunk_text),
                    "max_tokens": self.max_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "preserve_sentences": self.preserve_sentences
                })
                documents.append(doc_chunk)
        
        if not documents:
            raise ChunkerError("No valid chunks could be created from file content")
        
        return documents
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if self.preserve_sentences:
            return self._split_preserving_sentences(text)
        else:
            return self._split_by_tokens(text)
    
    def _split_by_tokens(self, text: str) -> List[str]:
        """
        Split text by token count without preserving sentence boundaries.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        tokens = self.tokenizer(text)
        chunks = []
        
        i = 0
        while i < len(tokens):
            # Determine chunk end
            chunk_end = min(i + self.max_tokens, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[i:chunk_end]
            chunk_text = ' '.join(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move to next chunk with overlap
            if chunk_end < len(tokens):
                i = chunk_end - self.overlap_tokens
            else:
                break
        
        return chunks
    
    def _split_preserving_sentences(self, text: str) -> List[str]:
        """
        Split text by token count while trying to preserve sentence boundaries.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds max_tokens, split it forcefully
            if sentence_tokens > self.max_tokens:
                # Add current chunk if it has content
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence
                sentence_chunks = self._split_by_tokens(sentence)
                chunks.extend(sentence_chunks)
                i += 1
                continue
            
            # Check if adding this sentence would exceed max_tokens
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                # Add current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_tokens = self.count_tokens(' '.join(current_chunk))
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Add final chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple rules.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of sentences
        """
        # Simple sentence splitting - can be improved with better NLP libraries
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        Get sentences for overlap from the end of current chunk.
        
        Args:
            sentences (List[str]): Current chunk sentences
            
        Returns:
            List[str]: Overlap sentences
        """
        if not sentences or self.overlap_tokens <= 0:
            return []
        
        overlap_sentences = []
        overlap_tokens = 0
        
        # Add sentences from the end until we reach overlap_tokens
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap_tokens:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def get_chunker_info(self) -> Dict[str, Any]:
        """
        Get information about the chunker configuration.
        
        Returns:
            Dict[str, Any]: Chunker information
        """
        info = super().get_chunker_info()
        info.update({
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "preserve_sentences": self.preserve_sentences,
            "tokenizer_type": "custom" if hasattr(self.tokenizer, '__name__') else "default"
        })
        return info


def create_base_length_chunker(
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    preserve_sentences: bool = True,
    **kwargs
) -> BaseLengthChunker:
    """
    Convenience function to create a BaseLengthChunker.
    
    Args:
        max_tokens (int): Maximum number of tokens per chunk
        overlap_tokens (int): Number of tokens to overlap between chunks
        preserve_sentences (bool): Whether to preserve sentence boundaries
        **kwargs: Additional configuration parameters
        
    Returns:
        BaseLengthChunker: Configured chunker instance
    """
    return BaseLengthChunker(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        preserve_sentences=preserve_sentences,
        **kwargs
    )
