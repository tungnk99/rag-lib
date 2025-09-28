"""
Semantic chunker implementation.

This module provides a semantic-based chunker that groups sentences
based on semantic similarity using sentence embeddings.
"""

from typing import List, Optional, Dict, Any, Callable, Union
import numpy as np
from dataclasses import dataclass
import re
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from ._base import BaseChunker, ChunkerError
from ..schemas.schema import Document, FileContent, Element, ElementType


@dataclass
class SemanticChunkConfig:
    """Configuration for semantic chunking."""
    similarity_threshold: float = 0.5
    max_tokens: int = 512
    min_tokens: int = 50
    max_sentences_per_chunk: int = 20
    min_sentences_per_chunk: int = 2
    overlap_sentences: int = 1
    use_clustering: bool = True
    clustering_method: str = "agglomerative"  # "agglomerative" or "threshold"


class SemanticChunkerError(ChunkerError):
    """Raised when semantic chunking fails."""
    pass


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that groups sentences based on semantic similarity.
    
    This chunker:
    1. Splits text into sentences
    2. Computes embeddings for each sentence
    3. Groups semantically similar sentences into chunks
    4. Ensures chunks meet size constraints
    
    Attributes:
        embedding_model: Function to compute sentence embeddings
        config: Semantic chunking configuration
        tokenizer: Function to tokenize text
    """
    
    def __init__(
        self,
        embedding_model: Optional[Callable[[List[str]], np.ndarray]] = None,
        config: Optional[SemanticChunkConfig] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        **kwargs
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            embedding_model: Function that takes list of sentences and returns embeddings
            config: Semantic chunking configuration
            tokenizer: Custom tokenizer function
            **kwargs: Additional configuration parameters
        """
        super().__init__(name="semantic_chunker", config=kwargs)
        
        # Set default config
        self.config = config or SemanticChunkConfig()
        
        # Set embedding model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = self._default_embedding_model
        
        # Set tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self._default_tokenizer
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate semantic chunker configuration."""
        if not (0 <= self.config.similarity_threshold <= 1):
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        if self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.config.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        
        if self.config.min_tokens >= self.config.max_tokens:
            raise ValueError("min_tokens must be less than max_tokens")
        
        if self.config.max_sentences_per_chunk <= 0:
            raise ValueError("max_sentences_per_chunk must be positive")
        
        if self.config.min_sentences_per_chunk <= 0:
            raise ValueError("min_sentences_per_chunk must be positive")
        
        if self.config.min_sentences_per_chunk > self.config.max_sentences_per_chunk:
            raise ValueError("min_sentences_per_chunk must be <= max_sentences_per_chunk")
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """Default whitespace-based tokenizer."""
        return text.split()
    
    def _default_embedding_model(self, sentences: List[str]) -> np.ndarray:
        """
        Default embedding model using simple word overlap similarity.
        
        Note: This is a fallback implementation. For production use,
        replace with proper sentence embeddings (SentenceTransformers, OpenAI, etc.)
        """
        print("Warning: Using default embedding model. Consider using SentenceTransformers or OpenAI embeddings for better results.")
        
        # Simple bag-of-words similarity
        embeddings = []
        
        # Get vocabulary
        all_words = set()
        for sentence in sentences:
            words = set(self.tokenizer(sentence.lower()))
            all_words.update(words)
        
        vocab = sorted(list(all_words))
        vocab_size = len(vocab)
        
        # Create word vectors
        for sentence in sentences:
            words = set(self.tokenizer(sentence.lower()))
            vector = np.zeros(vocab_size)
            
            for i, word in enumerate(vocab):
                if word in words:
                    vector[i] = 1.0
            
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            embeddings.append(vector)
        
        return np.array(embeddings)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer(text))
    
    def chunk_file_content(self, file_content: FileContent) -> List[Document]:
        """
        Split file content into document chunks using semantic similarity.
        
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
        
        try:
            # Perform semantic chunking
            chunks_data = self._semantic_split(text_content)
            
            # Create document chunks
            documents = []
            for i, chunk_data in enumerate(chunks_data):
                if chunk_data["text"].strip():
                    doc_chunk = self._create_document_chunk(
                        content=chunk_data["text"],
                        file_content=file_content,
                        chunk_index=i,
                        element_metadata=element_metadata
                    )
                    
                    # Add semantic chunking metadata
                    doc_chunk.metadata.update({
                        "chunk_method": "semantic",
                        "sentence_count": chunk_data["sentence_count"],
                        "avg_similarity": chunk_data.get("avg_similarity", 0.0),
                        "chunk_tokens": self.count_tokens(chunk_data["text"]),
                        "similarity_threshold": self.config.similarity_threshold,
                        "sentence_indices": chunk_data.get("sentence_indices", [])
                    })
                    
                    documents.append(doc_chunk)
            
            if not documents:
                raise SemanticChunkerError("No valid chunks could be created from file content")
            
            return documents
            
        except Exception as e:
            raise SemanticChunkerError(f"Semantic chunking failed: {str(e)}")
    
    def _semantic_split(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text using semantic similarity.
        
        Args:
            text: Text to split
            
        Returns:
            List of chunk data dictionaries
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= self.config.min_sentences_per_chunk:
            # Too few sentences to split semantically
            return [{
                "text": text,
                "sentence_count": len(sentences),
                "sentence_indices": list(range(len(sentences)))
            }]
        
        # Compute embeddings for sentences
        embeddings = self.embedding_model(sentences)
        
        # Group sentences based on similarity
        if self.config.use_clustering:
            sentence_groups = self._cluster_sentences(sentences, embeddings)
        else:
            sentence_groups = self._threshold_grouping(sentences, embeddings)
        
        # Convert sentence groups to chunks
        chunks_data = []
        for group in sentence_groups:
            chunk_text = " ".join(sentences[i] for i in group["indices"])
            
            # Validate chunk size
            chunk_tokens = self.count_tokens(chunk_text)
            
            # If chunk is too large, split it further
            if chunk_tokens > self.config.max_tokens:
                sub_chunks = self._split_large_chunk(group, sentences, embeddings)
                chunks_data.extend(sub_chunks)
            # If chunk is too small, try to merge with adjacent chunks
            elif chunk_tokens < self.config.min_tokens and chunks_data:
                # Try to merge with previous chunk
                prev_chunk = chunks_data[-1]
                merged_text = prev_chunk["text"] + " " + chunk_text
                merged_tokens = self.count_tokens(merged_text)
                
                if merged_tokens <= self.config.max_tokens:
                    chunks_data[-1]["text"] = merged_text
                    chunks_data[-1]["sentence_count"] += group["sentence_count"]
                    chunks_data[-1]["sentence_indices"].extend(group["indices"])
                else:
                    # Can't merge, add as separate chunk
                    chunks_data.append({
                        "text": chunk_text,
                        "sentence_count": group["sentence_count"],
                        "avg_similarity": group.get("avg_similarity", 0.0),
                        "sentence_indices": group["indices"]
                    })
            else:
                # Chunk size is good
                chunks_data.append({
                    "text": chunk_text,
                    "sentence_count": group["sentence_count"],
                    "avg_similarity": group.get("avg_similarity", 0.0),
                    "sentence_indices": group["indices"]
                })
        
        # Add overlap between chunks if configured
        if self.config.overlap_sentences > 0:
            chunks_data = self._add_sentence_overlap(chunks_data, sentences)
        
        return chunks_data
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Enhanced sentence splitting with better handling of abbreviations
        sentence_patterns = [
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+',  # Basic sentence endings
            r'(?<=\.)\s+(?=[A-Z])',  # Period followed by capital letter
            r'(?<=\!)\s+(?=[A-Z])',  # Exclamation followed by capital letter  
            r'(?<=\?)\s+(?=[A-Z])'   # Question mark followed by capital letter
        ]
        
        sentences = [text]
        
        for pattern in sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                split_sentences = re.split(pattern, sentence)
                new_sentences.extend(split_sentences)
            sentences = new_sentences
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # At least 3 words
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _cluster_sentences(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Cluster sentences using hierarchical clustering.
        
        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings
            
        Returns:
            List of sentence groups
        """
        if len(sentences) <= 1:
            return [{
                "indices": [0] if sentences else [],
                "sentence_count": len(sentences),
                "avg_similarity": 1.0
            }]
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Determine number of clusters
        n_clusters = max(1, min(
            len(sentences) // self.config.min_sentences_per_chunk,
            len(sentences) // 2
        ))
        
        try:
            # Perform clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group sentences by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            # Convert to groups
            groups = []
            for cluster_indices in clusters.values():
                # Calculate average similarity within cluster
                avg_similarity = 0.0
                if len(cluster_indices) > 1:
                    similarities = []
                    for i in range(len(cluster_indices)):
                        for j in range(i + 1, len(cluster_indices)):
                            sim = similarity_matrix[cluster_indices[i], cluster_indices[j]]
                            similarities.append(sim)
                    avg_similarity = np.mean(similarities) if similarities else 0.0
                else:
                    avg_similarity = 1.0
                
                groups.append({
                    "indices": sorted(cluster_indices),
                    "sentence_count": len(cluster_indices),
                    "avg_similarity": float(avg_similarity)
                })
            
            # Sort groups by first sentence index
            groups.sort(key=lambda g: g["indices"][0])
            
            return groups
            
        except Exception as e:
            # Fallback to threshold grouping if clustering fails
            print(f"Clustering failed: {e}. Falling back to threshold grouping.")
            return self._threshold_grouping(sentences, embeddings)
    
    def _threshold_grouping(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Group sentences using similarity threshold.
        
        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings
            
        Returns:
            List of sentence groups
        """
        if len(sentences) <= 1:
            return [{
                "indices": [0] if sentences else [],
                "sentence_count": len(sentences),
                "avg_similarity": 1.0
            }]
        
        groups = []
        current_group = [0]
        current_similarities = []
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = cosine_similarity(
                embeddings[i-1:i], 
                embeddings[i:i+1]
            )[0, 0]
            
            # Check if should add to current group
            should_add = (
                similarity >= self.config.similarity_threshold and
                len(current_group) < self.config.max_sentences_per_chunk
            )
            
            if should_add:
                current_group.append(i)
                current_similarities.append(similarity)
            else:
                # Finish current group
                avg_similarity = np.mean(current_similarities) if current_similarities else 1.0
                groups.append({
                    "indices": current_group.copy(),
                    "sentence_count": len(current_group),
                    "avg_similarity": float(avg_similarity)
                })
                
                # Start new group
                current_group = [i]
                current_similarities = []
        
        # Add final group
        if current_group:
            avg_similarity = np.mean(current_similarities) if current_similarities else 1.0
            groups.append({
                "indices": current_group,
                "sentence_count": len(current_group),
                "avg_similarity": float(avg_similarity)
            })
        
        return groups
    
    def _split_large_chunk(
        self, 
        group: Dict[str, Any], 
        sentences: List[str], 
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Split a large chunk into smaller chunks.
        
        Args:
            group: Sentence group that's too large
            sentences: All sentences
            embeddings: All sentence embeddings
            
        Returns:
            List of smaller chunks
        """
        indices = group["indices"]
        
        if len(indices) <= self.config.min_sentences_per_chunk:
            # Can't split further
            chunk_text = " ".join(sentences[i] for i in indices)
            return [{
                "text": chunk_text,
                "sentence_count": len(indices),
                "sentence_indices": indices
            }]
        
        # Split the group roughly in half
        mid_point = len(indices) // 2
        
        first_half = {
            "indices": indices[:mid_point],
            "sentence_count": mid_point
        }
        
        second_half = {
            "indices": indices[mid_point:],
            "sentence_count": len(indices) - mid_point
        }
        
        # Recursively check if these need further splitting
        result = []
        for sub_group in [first_half, second_half]:
            chunk_text = " ".join(sentences[i] for i in sub_group["indices"])
            chunk_tokens = self.count_tokens(chunk_text)
            
            if chunk_tokens > self.config.max_tokens:
                # Need to split further
                sub_chunks = self._split_large_chunk(sub_group, sentences, embeddings)
                result.extend(sub_chunks)
            else:
                result.append({
                    "text": chunk_text,
                    "sentence_count": sub_group["sentence_count"],
                    "sentence_indices": sub_group["indices"]
                })
        
        return result
    
    def _add_sentence_overlap(
        self, 
        chunks_data: List[Dict[str, Any]], 
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Add sentence overlap between chunks.
        
        Args:
            chunks_data: List of chunk data
            sentences: All sentences
            
        Returns:
            Chunks with overlap added
        """
        if len(chunks_data) <= 1 or self.config.overlap_sentences <= 0:
            return chunks_data
        
        overlapped_chunks = []
        
        for i, chunk_data in enumerate(chunks_data):
            chunk_text = chunk_data["text"]
            sentence_indices = chunk_data["sentence_indices"]
            
            # Add overlap from previous chunk
            if i > 0:
                prev_indices = chunks_data[i - 1]["sentence_indices"]
                overlap_count = min(self.config.overlap_sentences, len(prev_indices))
                
                if overlap_count > 0:
                    overlap_indices = prev_indices[-overlap_count:]
                    overlap_text = " ".join(sentences[j] for j in overlap_indices)
                    chunk_text = overlap_text + " " + chunk_text
                    sentence_indices = overlap_indices + sentence_indices
            
            # Update chunk data
            updated_chunk = chunk_data.copy()
            updated_chunk["text"] = chunk_text
            updated_chunk["sentence_indices"] = sentence_indices
            updated_chunk["sentence_count"] = len(sentence_indices)
            
            overlapped_chunks.append(updated_chunk)
        
        return overlapped_chunks
    
    def get_chunker_info(self) -> Dict[str, Any]:
        """Get information about the chunker configuration."""
        info = super().get_chunker_info()
        info.update({
            "similarity_threshold": self.config.similarity_threshold,
            "max_tokens": self.config.max_tokens,
            "min_tokens": self.config.min_tokens,
            "max_sentences_per_chunk": self.config.max_sentences_per_chunk,
            "min_sentences_per_chunk": self.config.min_sentences_per_chunk,
            "overlap_sentences": self.config.overlap_sentences,
            "use_clustering": self.config.use_clustering,
            "clustering_method": self.config.clustering_method
        })
        return info


def create_semantic_chunker(
    embedding_model: Optional[Callable[[List[str]], np.ndarray]] = None,
    similarity_threshold: float = 0.5,
    max_tokens: int = 512,
    min_tokens: int = 50,
    **kwargs
) -> SemanticChunker:
    """
    Convenience function to create a SemanticChunker.
    
    Args:
        embedding_model: Function to compute sentence embeddings
        similarity_threshold: Minimum similarity to group sentences
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk
        **kwargs: Additional configuration parameters
        
    Returns:
        SemanticChunker: Configured chunker instance
    """
    config = SemanticChunkConfig(
        similarity_threshold=similarity_threshold,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        **kwargs
    )
    
    return SemanticChunker(
        embedding_model=embedding_model,
        config=config
    )


def create_openai_semantic_chunker(
    api_key: str,
    model: str = "text-embedding-3-small",
    **kwargs
) -> SemanticChunker:
    """
    Create a semantic chunker using OpenAI embeddings.
    
    Args:
        api_key: OpenAI API key
        model: OpenAI embedding model name
        **kwargs: Additional configuration parameters
        
    Returns:
        SemanticChunker: Chunker with OpenAI embeddings
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai package is required for OpenAI semantic chunker")
    
    client = openai.OpenAI(api_key=api_key)
    
    def openai_embedding_model(sentences: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API."""
        response = client.embeddings.create(
            input=sentences,
            model=model
        )
        
        embeddings = []
        for item in response.data:
            embeddings.append(item.embedding)
        
        return np.array(embeddings)
    
    return create_semantic_chunker(
        embedding_model=openai_embedding_model,
        **kwargs
    )


def create_sentence_transformer_semantic_chunker(
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs
) -> SemanticChunker:
    """
    Create a semantic chunker using SentenceTransformers.
    
    Args:
        model_name: SentenceTransformers model name
        **kwargs: Additional configuration parameters
        
    Returns:
        SemanticChunker: Chunker with SentenceTransformers embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers package is required for SentenceTransformer semantic chunker")
    
    model = SentenceTransformer(model_name)
    
    def sentence_transformer_embedding_model(sentences: List[str]) -> np.ndarray:
        """Get embeddings from SentenceTransformers."""
        embeddings = model.encode(sentences)
        return np.array(embeddings)
    
    return create_semantic_chunker(
        embedding_model=sentence_transformer_embedding_model,
        **kwargs
    )
