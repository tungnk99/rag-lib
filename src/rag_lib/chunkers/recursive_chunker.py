"""
Recursive chunker implementation.

This module provides a hierarchical chunker that recursively splits documents
into smaller chunks while preserving document structure and relationships.
"""

from typing import List, Optional, Dict, Any, Callable, Tuple
import re
from dataclasses import dataclass
from ._base import BaseChunker, ChunkerError
from ..schemas.schema import Document, FileContent, Element, ElementType


@dataclass
class ChunkLevel:
    """Represents a level in the chunking hierarchy."""
    name: str
    separators: List[str]
    max_tokens: int
    min_tokens: int = 10
    keep_separator: bool = True


class RecursiveChunker(BaseChunker):
    """
    Recursive hierarchical chunker that splits documents at multiple levels.
    
    This chunker processes documents in a hierarchical manner:
    1. First tries to split by major separators (sections, paragraphs)
    2. If chunks are still too large, recursively splits by smaller separators
    3. Maintains parent-child relationships between chunks
    
    Attributes:
        chunk_levels (List[ChunkLevel]): Hierarchy of chunking levels
        tokenizer (Callable): Function to tokenize text
        overlap_tokens (int): Number of tokens to overlap between chunks
        create_parent_chunks (bool): Whether to create parent chunks for hierarchy
    """
    
    def __init__(
        self,
        chunk_levels: Optional[List[ChunkLevel]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        overlap_tokens: int = 50,
        create_parent_chunks: bool = True,
        **kwargs
    ):
        """
        Initialize the recursive chunker.
        
        Args:
            chunk_levels: Hierarchy of chunking levels. If None, uses default levels
            tokenizer: Custom tokenizer function
            overlap_tokens: Number of tokens to overlap between chunks
            create_parent_chunks: Whether to create parent chunks for hierarchy
            **kwargs: Additional configuration parameters
        """
        super().__init__(name="recursive_chunker", config=kwargs)
        
        # Set default chunk levels if not provided
        if chunk_levels is None:
            self.chunk_levels = self._get_default_chunk_levels()
        else:
            self.chunk_levels = chunk_levels
        
        self.overlap_tokens = overlap_tokens
        self.create_parent_chunks = create_parent_chunks
        
        # Set up tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self._default_tokenizer
        
        self._validate_config()
    
    def _get_default_chunk_levels(self) -> List[ChunkLevel]:
        """Get default chunking levels hierarchy."""
        return [
            # Level 1: Major sections (headers, double newlines)
            ChunkLevel(
                name="section",
                separators=["\n\n\n", "\n\n", "# ", "## ", "### "],
                max_tokens=2048,
                min_tokens=100,
                keep_separator=True
            ),
            # Level 2: Paragraphs and sentences
            ChunkLevel(
                name="paragraph", 
                separators=["\n", ". ", "! ", "? "],
                max_tokens=512,
                min_tokens=50,
                keep_separator=True
            ),
            # Level 3: Clauses and phrases
            ChunkLevel(
                name="sentence",
                separators=[", ", "; ", ": ", " - "],
                max_tokens=128,
                min_tokens=10,
                keep_separator=False
            ),
            # Level 4: Words (fallback)
            ChunkLevel(
                name="word",
                separators=[" "],
                max_tokens=64,
                min_tokens=1,
                keep_separator=False
            )
        ]
    
    def _validate_config(self) -> None:
        """Validate chunker configuration."""
        if not self.chunk_levels:
            raise ValueError("At least one chunk level must be provided")
        
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        
        # Validate chunk levels are in descending order of max_tokens
        for i in range(len(self.chunk_levels) - 1):
            if self.chunk_levels[i].max_tokens <= self.chunk_levels[i + 1].max_tokens:
                raise ValueError("Chunk levels must be in descending order of max_tokens")
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """Default whitespace-based tokenizer."""
        return text.split()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer(text))
    
    def chunk_file_content(self, file_content: FileContent) -> List[Document]:
        """
        Split file content into document chunks using recursive chunking.
        
        Args:
            file_content: File content to be chunked
            
        Returns:
            List of document chunks
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
        
        # Perform recursive chunking
        chunks_data = self._recursive_split(text_content, level=0)
        
        # Create document chunks
        documents = []
        chunk_index = 0
        
        for chunk_data in chunks_data:
            if chunk_data["text"].strip():
                doc_chunk = self._create_document_chunk(
                    content=chunk_data["text"],
                    file_content=file_content,
                    chunk_index=chunk_index,
                    element_metadata=element_metadata
                )
                
                # Add recursive chunking metadata
                doc_chunk.metadata.update({
                    "chunk_level": chunk_data["level"],
                    "chunk_level_name": chunk_data["level_name"],
                    "parent_chunk_id": chunk_data.get("parent_id"),
                    "child_chunk_ids": chunk_data.get("child_ids", []),
                    "chunk_tokens": self.count_tokens(chunk_data["text"]),
                    "max_tokens_for_level": chunk_data["max_tokens"],
                    "separators_used": chunk_data.get("separators_used", [])
                })
                
                documents.append(doc_chunk)
                chunk_index += 1
        
        if not documents:
            raise ChunkerError("No valid chunks could be created from file content")
        
        return documents
    
    def _recursive_split(
        self, 
        text: str, 
        level: int = 0, 
        parent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively split text at different levels.
        
        Args:
            text: Text to split
            level: Current chunking level (0-based)
            parent_id: ID of parent chunk
            
        Returns:
            List of chunk data dictionaries
        """
        if level >= len(self.chunk_levels):
            # Reached maximum depth, return as single chunk
            return [{
                "text": text,
                "level": level - 1,
                "level_name": self.chunk_levels[level - 1].name if level > 0 else "max_depth",
                "parent_id": parent_id,
                "max_tokens": self.chunk_levels[level - 1].max_tokens if level > 0 else 0,
                "separators_used": []
            }]
        
        current_level = self.chunk_levels[level]
        token_count = self.count_tokens(text)
        
        # If text is within token limit, return as single chunk
        if token_count <= current_level.max_tokens:
            return [{
                "text": text,
                "level": level,
                "level_name": current_level.name,
                "parent_id": parent_id,
                "max_tokens": current_level.max_tokens,
                "separators_used": []
            }]
        
        # Try to split using separators at current level
        chunks = self._split_by_separators(text, current_level.separators, current_level.keep_separator)
        
        # If splitting didn't help much, move to next level
        if len(chunks) <= 1:
            return self._recursive_split(text, level + 1, parent_id)
        
        result_chunks = []
        separators_used = []
        
        for chunk_text in chunks:
            chunk_tokens = self.count_tokens(chunk_text)
            
            # If chunk is still too large, recursively split
            if chunk_tokens > current_level.max_tokens:
                sub_chunks = self._recursive_split(chunk_text, level + 1, parent_id)
                result_chunks.extend(sub_chunks)
            # If chunk is too small, try to merge with adjacent chunks
            elif chunk_tokens < current_level.min_tokens and result_chunks:
                # Try to merge with previous chunk
                prev_chunk = result_chunks[-1]
                merged_text = prev_chunk["text"] + " " + chunk_text
                merged_tokens = self.count_tokens(merged_text)
                
                if merged_tokens <= current_level.max_tokens:
                    result_chunks[-1]["text"] = merged_text
                    result_chunks[-1]["separators_used"].extend(separators_used)
                else:
                    # Can't merge, add as separate chunk
                    result_chunks.append({
                        "text": chunk_text,
                        "level": level,
                        "level_name": current_level.name,
                        "parent_id": parent_id,
                        "max_tokens": current_level.max_tokens,
                        "separators_used": separators_used.copy()
                    })
            else:
                # Chunk size is good, add it
                result_chunks.append({
                    "text": chunk_text,
                    "level": level,
                    "level_name": current_level.name,
                    "parent_id": parent_id,
                    "max_tokens": current_level.max_tokens,
                    "separators_used": separators_used.copy()
                })
        
        # Add overlap between chunks if configured
        if self.overlap_tokens > 0 and len(result_chunks) > 1:
            result_chunks = self._add_overlap_to_chunks(result_chunks)
        
        return result_chunks
    
    def _split_by_separators(
        self, 
        text: str, 
        separators: List[str], 
        keep_separator: bool = True
    ) -> List[str]:
        """
        Split text using a list of separators in priority order.
        
        Args:
            text: Text to split
            separators: List of separators in priority order
            keep_separator: Whether to keep separator in chunks
            
        Returns:
            List of text chunks
        """
        chunks = [text]
        
        for separator in separators:
            new_chunks = []
            for chunk in chunks:
                if separator in chunk:
                    split_chunks = chunk.split(separator)
                    
                    if keep_separator and len(split_chunks) > 1:
                        # Re-add separator to chunks (except last one)
                        processed_chunks = []
                        for i, split_chunk in enumerate(split_chunks[:-1]):
                            processed_chunks.append(split_chunk + separator)
                        processed_chunks.append(split_chunks[-1])
                        new_chunks.extend(processed_chunks)
                    else:
                        new_chunks.extend(split_chunks)
                else:
                    new_chunks.append(chunk)
            
            chunks = [chunk for chunk in new_chunks if chunk.strip()]
            
            # If we got good splits, stop trying other separators
            if len(chunks) > 1:
                break
        
        return chunks
    
    def _add_overlap_to_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add overlap between adjacent chunks.
        
        Args:
            chunks: List of chunk data dictionaries
            
        Returns:
            List of chunks with overlap added
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            
            # Add overlap from previous chunk
            if i > 0 and self.overlap_tokens > 0:
                prev_chunk_text = chunks[i - 1]["text"]
                prev_tokens = self.tokenizer(prev_chunk_text)
                
                if len(prev_tokens) >= self.overlap_tokens:
                    overlap_tokens = prev_tokens[-self.overlap_tokens:]
                    overlap_text = " ".join(overlap_tokens)
                    chunk_text = overlap_text + " " + chunk_text
            
            # Update chunk data
            updated_chunk = chunk.copy()
            updated_chunk["text"] = chunk_text
            overlapped_chunks.append(updated_chunk)
        
        return overlapped_chunks
    
    def get_chunker_info(self) -> Dict[str, Any]:
        """Get information about the chunker configuration."""
        info = super().get_chunker_info()
        info.update({
            "chunk_levels": [
                {
                    "name": level.name,
                    "separators": level.separators,
                    "max_tokens": level.max_tokens,
                    "min_tokens": level.min_tokens
                }
                for level in self.chunk_levels
            ],
            "overlap_tokens": self.overlap_tokens,
            "create_parent_chunks": self.create_parent_chunks,
            "total_levels": len(self.chunk_levels)
        })
        return info


def create_recursive_chunker(
    chunk_levels: Optional[List[ChunkLevel]] = None,
    overlap_tokens: int = 50,
    create_parent_chunks: bool = True,
    **kwargs
) -> RecursiveChunker:
    """
    Convenience function to create a RecursiveChunker.
    
    Args:
        chunk_levels: Custom chunk levels hierarchy
        overlap_tokens: Number of tokens to overlap between chunks
        create_parent_chunks: Whether to create parent chunks
        **kwargs: Additional configuration parameters
        
    Returns:
        RecursiveChunker: Configured chunker instance
    """
    return RecursiveChunker(
        chunk_levels=chunk_levels,
        overlap_tokens=overlap_tokens,
        create_parent_chunks=create_parent_chunks,
        **kwargs
    )


def create_code_recursive_chunker(**kwargs) -> RecursiveChunker:
    """Create a recursive chunker optimized for code files."""
    code_levels = [
        ChunkLevel(
            name="class_function",
            separators=["class ", "def ", "function ", "async def "],
            max_tokens=1024,
            min_tokens=50,
            keep_separator=True
        ),
        ChunkLevel(
            name="block",
            separators=["\n\n", "{\n", "}\n"],
            max_tokens=256,
            min_tokens=20,
            keep_separator=True
        ),
        ChunkLevel(
            name="statement",
            separators=["\n", ";"],
            max_tokens=64,
            min_tokens=5,
            keep_separator=True
        ),
        ChunkLevel(
            name="token",
            separators=[" ", "\t"],
            max_tokens=16,
            min_tokens=1,
            keep_separator=False
        )
    ]
    
    return RecursiveChunker(chunk_levels=code_levels, **kwargs)


def create_academic_recursive_chunker(**kwargs) -> RecursiveChunker:
    """Create a recursive chunker optimized for academic papers."""
    academic_levels = [
        ChunkLevel(
            name="section",
            separators=["# ", "## ", "### ", "#### ", "\n\nAbstract", "\n\nIntroduction", "\n\nConclusion"],
            max_tokens=3072,
            min_tokens=200,
            keep_separator=True
        ),
        ChunkLevel(
            name="subsection",
            separators=["\n\n", ". ", "! ", "? "],
            max_tokens=768,
            min_tokens=100,
            keep_separator=True
        ),
        ChunkLevel(
            name="sentence",
            separators=[", ", "; ", ": "],
            max_tokens=192,
            min_tokens=20,
            keep_separator=True
        ),
        ChunkLevel(
            name="phrase",
            separators=[" "],
            max_tokens=64,
            min_tokens=5,
            keep_separator=False
        )
    ]
    
    return RecursiveChunker(chunk_levels=academic_levels, **kwargs)
