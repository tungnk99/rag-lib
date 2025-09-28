"""
Data pipeline for processing documents through the RAG system.

This module provides a complete pipeline that processes files through:
Reader -> Chunker -> Document Store (via retriever interface)
"""

from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
import time
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..readers._base import BaseReader, MultiFormatReader
from ..readers._factory import ReaderFactory, get_default_factory
from ..chunkers._base import BaseChunker
from ..chunkers._factory import ChunkerFactory, create_chunker
from ..document_stores._base import BaseDocumentStore
from ..schemas.schema import Document, FileContent


@dataclass
class PipelineStats:
    """
    Statistics for pipeline execution.
    
    Attributes:
        files_processed (int): Number of files successfully processed
        files_failed (int): Number of files that failed processing
        total_documents_created (int): Total number of document chunks created
        total_processing_time (float): Total time taken for processing
        average_file_time (float): Average time per file
        stage_times (Dict[str, float]): Time taken by each pipeline stage
    """
    files_processed: int = 0
    files_failed: int = 0
    total_documents_created: int = 0
    total_processing_time: float = 0.0
    average_file_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    def add_file_result(self, success: bool, processing_time: float, documents_count: int = 0):
        """Add result from processing a single file."""
        if success:
            self.files_processed += 1
            self.total_documents_created += documents_count
        else:
            self.files_failed += 1
        
        self.total_processing_time += processing_time
        total_files = self.files_processed + self.files_failed
        if total_files > 0:
            self.average_file_time = self.total_processing_time / total_files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary representation."""
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "total_documents_created": self.total_documents_created,
            "total_processing_time": self.total_processing_time,
            "average_file_time": self.average_file_time,
            "stage_times": self.stage_times
        }


@dataclass
class PipelineResult:
    """
    Result of pipeline execution.
    
    Attributes:
        success (bool): Whether the pipeline completed successfully
        stats (PipelineStats): Execution statistics
        processed_files (List[str]): List of successfully processed file paths
        failed_files (List[Dict[str, Any]]): List of failed files with error info
        metadata (Dict[str, Any]): Additional metadata about the execution
    """
    success: bool
    stats: PipelineStats
    processed_files: List[str] = field(default_factory=list)
    failed_files: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "success": self.success,
            "stats": self.stats.to_dict(),
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "metadata": self.metadata
        }


@dataclass
class PipelineConfig:
    """
    Configuration for data pipeline creation.
    
    Attributes:
        reader (Optional[Dict[str, Any]]): Reader configuration
        chunker (Dict[str, Any]): Chunker configuration
        document_store (Optional[Dict[str, Any]]): Document store configuration
        enable_progress (bool): Whether to enable progress tracking
        batch_size (int): Number of files to process in each batch
        logger_config (Optional[Dict[str, Any]]): Logger configuration
    """
    chunker: Dict[str, Any]
    reader: Optional[Dict[str, Any]] = None
    document_store: Optional[Dict[str, Any]] = None
    enable_progress: bool = True
    batch_size: int = 10
    logger_config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create PipelineConfig from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "reader": self.reader,
            "chunker": self.chunker,
            "document_store": self.document_store,
            "enable_progress": self.enable_progress,
            "batch_size": self.batch_size,
            "logger_config": self.logger_config
        }


class DataPipelineError(Exception):
    """Base exception for data pipeline operations."""
    pass


class DataPipeline:
    """
    Complete data processing pipeline for RAG system.
    
    Processes files through: Reader -> Chunker -> Document Store
    
    Attributes:
        reader (BaseReader): Reader for processing files
        chunker (BaseChunker): Chunker for splitting content
        document_store (BaseDocumentStore): Store for saving document chunks
        logger (logging.Logger): Logger for pipeline operations
    """
    
    def __init__(
        self,
        reader: Optional[BaseReader] = None,
        chunker: Optional[BaseChunker] = None,
        document_store: Optional[BaseDocumentStore] = None,
        logger: Optional[logging.Logger] = None,
        enable_progress: bool = True,
        batch_size: int = 10,
        reader_factory: Optional[ReaderFactory] = None,
        chunker_factory: Optional[ChunkerFactory] = None
    ):
        """
        Initialize the data pipeline.
        
        Args:
            reader (Optional[BaseReader]): File reader. If None, uses default factory
            chunker (Optional[BaseChunker]): Document chunker. Must be provided for processing
            document_store (Optional[BaseDocumentStore]): Document store. Must be provided for processing
            logger (Optional[logging.Logger]): Logger for pipeline operations
            enable_progress (bool): Whether to enable progress tracking
            batch_size (int): Number of files to process in each batch
            reader_factory (Optional[ReaderFactory]): Factory for creating readers
            chunker_factory (Optional[ChunkerFactory]): Factory for creating chunkers
        """
        self.reader = reader
        self.chunker = chunker
        self.document_store = document_store
        self.logger = logger or self._setup_default_logger()
        self.enable_progress = enable_progress
        self.batch_size = batch_size
        self.reader_factory = reader_factory or get_default_factory()
        self.chunker_factory = chunker_factory or ChunkerFactory()
        
        # Setup default reader if none provided
        if self.reader is None:
            self._setup_default_reader()
    
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger for the pipeline."""
        logger = logging.getLogger("DataPipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_default_reader(self):
        """Setup default reader using factory."""
        # Use factory to get a reader that can handle files automatically
        self.reader = MultiFormatReader()
        
        # Setup readers from factory
        for reader_name, reader_info in self.reader_factory.list_readers().items():
            try:
                reader_instance = self.reader_factory._registry[reader_name].create_reader()
                self.reader.register_reader(reader_instance)
                self.logger.debug(f"Registered reader: {reader_name}")
            except Exception as e:
                self.logger.warning(f"Could not register {reader_name}: {e}")
    
    def validate_pipeline(self) -> None:
        """
        Validate that pipeline components are properly configured.
        
        Raises:
            DataPipelineError: If pipeline is not properly configured
        """
        if self.chunker is None:
            raise DataPipelineError("Chunker is required for pipeline operation")
        
        if self.document_store is None:
            raise DataPipelineError("Document store is required for pipeline operation")
        
        if isinstance(self.reader, MultiFormatReader) and not self.reader.readers:
            raise DataPipelineError("No readers available for file processing")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline.
        
        Args:
            file_path (str): Path to the file to process
            
        Returns:
            Dict[str, Any]: Processing result with stats and document IDs
            
        Raises:
            DataPipelineError: If processing fails
        """
        start_time = time.time()
        stage_times = {}
        
        try:
            # Validate pipeline before processing
            self.validate_pipeline()
            
            self.logger.info(f"Starting pipeline processing for: {file_path}")
            
            # Stage 1: Read file content
            read_start = time.time()
            self.logger.debug(f"Reading file: {file_path}")
            
            file_content = self.reader.read(file_path)
            
            read_time = time.time() - read_start
            stage_times["read"] = read_time
            
            self.logger.info(
                f"File read completed in {read_time:.2f}s. "
                f"Extracted {len(file_content.elements)} elements"
            )
            
            # Stage 2: Chunk the content
            chunk_start = time.time()
            self.logger.debug("Chunking file content")
            
            document_chunks = self.chunker.chunk_file_content(file_content)
            
            chunk_time = time.time() - chunk_start
            stage_times["chunk"] = chunk_time
            
            self.logger.info(
                f"Chunking completed in {chunk_time:.2f}s. "
                f"Created {len(document_chunks)} chunks"
            )
            
            # Stage 3: Store documents
            store_start = time.time()
            self.logger.debug(f"Storing {len(document_chunks)} documents")
            
            stored_ids = []
            for doc in document_chunks:
                doc_id = self.document_store.add_document(doc)
                stored_ids.append(doc_id)
            
            store_time = time.time() - store_start
            stage_times["store"] = store_time
            
            self.logger.info(
                f"Storage completed in {store_time:.2f}s. "
                f"Stored {len(stored_ids)} documents"
            )
            
            # Calculate total time
            total_time = time.time() - start_time
            
            result = {
                "success": True,
                "file_path": file_path,
                "documents_created": len(document_chunks),
                "document_ids": stored_ids,
                "processing_time": total_time,
                "stage_times": stage_times,
                "file_metadata": file_content.metadata
            }
            
            self.logger.info(
                f"Pipeline completed successfully for {file_path} in {total_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Pipeline failed for {file_path}: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": total_time,
                "stage_times": stage_times
            }
    
    def process_files(
        self,
        file_paths: List[str],
        continue_on_error: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> PipelineResult:
        """
        Process multiple files through the pipeline.
        
        Args:
            file_paths (List[str]): List of file paths to process
            continue_on_error (bool): Whether to continue processing if a file fails
            progress_callback (Optional[Callable]): Callback for progress updates
            
        Returns:
            PipelineResult: Complete processing results
        """
        start_time = time.time()
        stats = PipelineStats()
        processed_files = []
        failed_files = []
        
        self.logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        try:
            # Validate pipeline before batch processing
            self.validate_pipeline()
            
            for i, file_path in enumerate(file_paths):
                try:
                    # Process single file
                    result = self.process_file(file_path)
                    
                    if result["success"]:
                        processed_files.append(file_path)
                        stats.add_file_result(
                            success=True,
                            processing_time=result["processing_time"],
                            documents_count=result["documents_created"]
                        )
                        
                        # Update stage times
                        for stage, stage_time in result["stage_times"].items():
                            if stage not in stats.stage_times:
                                stats.stage_times[stage] = 0
                            stats.stage_times[stage] += stage_time
                        
                    else:
                        failed_files.append({
                            "file_path": file_path,
                            "error": result["error"],
                            "error_type": result["error_type"]
                        })
                        stats.add_file_result(
                            success=False,
                            processing_time=result["processing_time"]
                        )
                        
                        if not continue_on_error:
                            raise DataPipelineError(f"Processing failed for {file_path}: {result['error']}")
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(i + 1, len(file_paths))
                    
                    if self.enable_progress and (i + 1) % max(1, len(file_paths) // 10) == 0:
                        progress = ((i + 1) / len(file_paths)) * 100
                        self.logger.info(f"Progress: {progress:.1f}% ({i + 1}/{len(file_paths)} files)")
                
                except Exception as e:
                    self.logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                    failed_files.append({
                        "file_path": file_path,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    stats.add_file_result(success=False, processing_time=0)
                    
                    if not continue_on_error:
                        raise
            
            # Calculate final stats
            total_time = time.time() - start_time
            success = len(failed_files) == 0
            
            result = PipelineResult(
                success=success,
                stats=stats,
                processed_files=processed_files,
                failed_files=failed_files,
                metadata={
                    "total_execution_time": total_time,
            "pipeline_config": {
                "reader": {
                    "type": type(self.reader).__name__,
                    "info": getattr(self.reader, "get_reader_info", lambda: {})()
                },
                "chunker": {
                    "type": type(self.chunker).__name__,
                    "info": self.chunker.get_chunker_info() if self.chunker else None
                },
                "document_store": {
                    "type": type(self.document_store).__name__,
                    "info": getattr(self.document_store, "get_store_info", lambda: {})()
                }
            },
                    "execution_timestamp": datetime.now().isoformat()
                }
            )
            
            # Log final results
            self.logger.info(
                f"Batch processing completed in {total_time:.2f}s. "
                f"Success: {stats.files_processed}/{len(file_paths)} files, "
                f"Failed: {stats.files_failed}, "
                f"Documents created: {stats.total_documents_created}"
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Batch processing failed: {str(e)}")
            
            return PipelineResult(
                success=False,
                stats=stats,
                processed_files=processed_files,
                failed_files=failed_files,
                metadata={
                    "total_execution_time": total_time,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "execution_timestamp": datetime.now().isoformat()
                }
            )
    
    def process_directory(
        self,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        **kwargs
    ) -> PipelineResult:
        """
        Process all files in a directory through the pipeline.
        
        Args:
            directory_path (str): Path to directory to process
            file_patterns (Optional[List[str]]): File patterns to match (e.g., ["*.txt", "*.pdf"])
            recursive (bool): Whether to search subdirectories
            **kwargs: Additional arguments for process_files
            
        Returns:
            PipelineResult: Processing results
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise DataPipelineError(f"Directory does not exist: {directory_path}")
        
        if not directory.is_dir():
            raise DataPipelineError(f"Path is not a directory: {directory_path}")
        
        # Find files to process
        file_paths = []
        
        if file_patterns:
            for pattern in file_patterns:
                if recursive:
                    files = directory.rglob(pattern)
                else:
                    files = directory.glob(pattern)
                file_paths.extend([str(f) for f in files if f.is_file()])
        else:
            # Process all files
            if recursive:
                files = directory.rglob("*")
            else:
                files = directory.glob("*")
            file_paths = [str(f) for f in files if f.is_file()]
        
        self.logger.info(f"Found {len(file_paths)} files to process in {directory_path}")
        
        if not file_paths:
            self.logger.warning(f"No files found in directory: {directory_path}")
            return PipelineResult(
                success=True,
                stats=PipelineStats(),
                metadata={"message": "No files found to process"}
            )
        
        return self.process_files(file_paths, **kwargs)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dict[str, Any]: Pipeline configuration info
        """
        return {
            "reader": {
                "type": type(self.reader).__name__,
                "info": getattr(self.reader, "get_reader_info", lambda: {})()
            },
            "chunker": {
                "type": type(self.chunker).__name__ if self.chunker else None,
                "info": self.chunker.get_chunker_info() if self.chunker else None
            },
            "document_store": {
                "type": type(self.document_store).__name__ if self.document_store else None,
                "info": getattr(self.document_store, "get_store_info", lambda: {})() if self.document_store else None
            },
            "batch_size": self.batch_size,
            "enable_progress": self.enable_progress,
            "factories": {
                "reader_factory": type(self.reader_factory).__name__,
                "chunker_factory": type(self.chunker_factory).__name__
            }
        }
    
    @classmethod
    def from_config(
        cls,
        config: Union[Dict[str, Any], PipelineConfig],
        document_store: Optional[BaseDocumentStore] = None
    ) -> 'DataPipeline':
        """
        Create a DataPipeline from configuration.
        
        Args:
            config (Union[Dict[str, Any], PipelineConfig]): Pipeline configuration
            document_store (Optional[BaseDocumentStore]): Document store instance
            
        Returns:
            DataPipeline: Configured pipeline instance
            
        Examples:
            >>> config = {
            ...     "reader": {"type": "auto"},  # Use factory auto-detection
            ...     "chunker": {
            ...         "type": "recursive",
            ...         "overlap_tokens": 50
            ...     },
            ...     "enable_progress": True,
            ...     "batch_size": 5
            ... }
            >>> pipeline = DataPipeline.from_config(config, document_store)
        """
        # Convert dict to PipelineConfig if needed
        if isinstance(config, dict):
            config = PipelineConfig.from_dict(config)
        
        # Create factories
        reader_factory = get_default_factory()
        chunker_factory = ChunkerFactory()
        
        # Create reader
        reader = None
        if config.reader:
            reader_type = config.reader.get("type", "auto")
            if reader_type != "auto":
                # Use specific reader from factory
                reader_config = config.reader.copy()
                reader_config.pop("type", None)
                try:
                    reader = reader_factory.get_reader_for_file(
                        "dummy." + reader_type,  # Use extension to trigger reader
                        **reader_config
                    )
                except Exception:
                    # Fallback to auto reader
                    reader = None
        
        # Create chunker
        chunker = None
        if config.chunker:
            chunker_config = config.chunker.copy()
            chunker = chunker_factory.create_chunker(
                chunker_config.pop("type"),
                chunker_config
            )
        
        # Setup logger if configured
        logger = None
        if config.logger_config:
            logger = cls._setup_logger_from_config(config.logger_config)
        
        # Create pipeline
        return cls(
            reader=reader,
            chunker=chunker,
            document_store=document_store,
            logger=logger,
            enable_progress=config.enable_progress,
            batch_size=config.batch_size,
            reader_factory=reader_factory,
            chunker_factory=chunker_factory
        )
    
    @staticmethod
    def _setup_logger_from_config(logger_config: Dict[str, Any]) -> logging.Logger:
        """Setup logger from configuration."""
        logger_name = logger_config.get("name", "DataPipeline")
        logger = logging.getLogger(logger_name)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Set level
        level = logger_config.get("level", "INFO")
        logger.setLevel(getattr(logging, level.upper()))
        
        # Add handler
        handler_type = logger_config.get("handler", "stream")
        if handler_type == "stream":
            handler = logging.StreamHandler()
        elif handler_type == "file":
            filename = logger_config.get("filename", "pipeline.log")
            handler = logging.FileHandler(filename)
        else:
            handler = logging.StreamHandler()
        
        # Set formatter
        format_str = logger_config.get(
            "format", 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger


# Convenience functions for creating pipelines

def create_data_pipeline(
    reader: Optional[BaseReader] = None,
    chunker: Optional[BaseChunker] = None,
    document_store: Optional[BaseDocumentStore] = None,
    **kwargs
) -> DataPipeline:
    """
    Convenience function to create a data pipeline.
    
    Args:
        reader (Optional[BaseReader]): File reader
        chunker (Optional[BaseChunker]): Document chunker
        document_store (Optional[BaseDocumentStore]): Document store
        **kwargs: Additional pipeline configuration
        
    Returns:
        DataPipeline: Configured pipeline
    """
    return DataPipeline(
        reader=reader,
        chunker=chunker,
        document_store=document_store,
        **kwargs
    )


def create_simple_pipeline(
    chunker: BaseChunker,
    document_store: BaseDocumentStore,
    **kwargs
) -> DataPipeline:
    """
    Create a simple pipeline with default reader using factory.
    
    Args:
        chunker (BaseChunker): Document chunker
        document_store (BaseDocumentStore): Document store
        **kwargs: Additional pipeline configuration
        
    Returns:
        DataPipeline: Configured pipeline with factory-based reader
    """
    return DataPipeline(
        reader=None,  # Will use factory to setup readers
        chunker=chunker,
        document_store=document_store,
        **kwargs
    )


def create_pipeline_from_config(
    config: Union[Dict[str, Any], PipelineConfig],
    document_store: BaseDocumentStore,
    **kwargs
) -> DataPipeline:
    """
    Create a pipeline from configuration dictionary or PipelineConfig.
    
    Args:
        config (Union[Dict[str, Any], PipelineConfig]): Pipeline configuration
        document_store (BaseDocumentStore): Document store instance
        **kwargs: Additional pipeline configuration to override
        
    Returns:
        DataPipeline: Configured pipeline
        
    Examples:
        >>> config = {
        ...     "chunker": {
        ...         "type": "recursive",
        ...         "overlap_tokens": 50
        ...     },
        ...     "enable_progress": True
        ... }
        >>> pipeline = create_pipeline_from_config(config, document_store)
    """
    # Override config with kwargs if provided
    if isinstance(config, dict):
        config = config.copy()
        config.update(kwargs)
        config = PipelineConfig.from_dict(config)
    else:
        # Update PipelineConfig fields with kwargs
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = PipelineConfig.from_dict(config_dict)
    
    return DataPipeline.from_config(config, document_store)


def create_pipeline_with_chunker_type(
    chunker_type: str,
    document_store: BaseDocumentStore,
    chunker_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> DataPipeline:
    """
    Create a pipeline with a specific chunker type using factory.
    
    Args:
        chunker_type (str): Type of chunker to create
        document_store (BaseDocumentStore): Document store instance
        chunker_config (Optional[Dict[str, Any]]): Chunker configuration
        **kwargs: Additional pipeline configuration
        
    Returns:
        DataPipeline: Configured pipeline
        
    Examples:
        >>> # Create pipeline with semantic chunker
        >>> pipeline = create_pipeline_with_chunker_type(
        ...     "semantic_sentence_transformer",
        ...     document_store,
        ...     {"similarity_threshold": 0.7}
        ... )
    """
    chunker_factory = ChunkerFactory()
    chunker = chunker_factory.create_chunker(chunker_type, chunker_config or {})
    
    return DataPipeline(
        reader=None,  # Use factory default
        chunker=chunker,
        document_store=document_store,
        **kwargs
    )


def get_available_chunker_types() -> List[str]:
    """Get list of available chunker types from factory."""
    factory = ChunkerFactory()
    return factory.get_available_types()


def get_available_reader_types() -> Dict[str, Any]:
    """Get information about available reader types from factory."""
    factory = get_default_factory()
    return factory.get_factory_info()


def create_example_configs() -> Dict[str, Dict[str, Any]]:
    """
    Create example pipeline configurations for different use cases.
    
    Returns:
        Dict[str, Dict[str, Any]]: Example configurations
    """
    return {
        "simple_text_processing": {
            "chunker": {
                "type": "length",
                "max_tokens": 512,
                "overlap_tokens": 50,
                "preserve_sentences": True
            },
            "enable_progress": True,
            "batch_size": 10
        },
        "hierarchical_documents": {
            "chunker": {
                "type": "recursive",
                "overlap_tokens": 50,
                "create_parent_chunks": True
            },
            "enable_progress": True,
            "batch_size": 5
        },
        "code_files": {
            "chunker": {
                "type": "recursive_code",
                "overlap_tokens": 25,
                "create_parent_chunks": True
            },
            "enable_progress": True,
            "batch_size": 5
        },
        "academic_papers": {
            "chunker": {
                "type": "recursive_academic",
                "overlap_tokens": 75,
                "create_parent_chunks": True
            },
            "enable_progress": True,
            "batch_size": 3
        },
        "semantic_chunking": {
            "chunker": {
                "type": "semantic_sentence_transformer",
                "model_name": "all-MiniLM-L6-v2",
                "similarity_threshold": 0.6,
                "max_tokens": 512,
                "min_tokens": 50
            },
            "enable_progress": True,
            "batch_size": 5
        },
        "high_quality_semantic": {
            "chunker": {
                "type": "semantic_sentence_transformer",
                "model_name": "all-mpnet-base-v2",
                "similarity_threshold": 0.7,
                "max_tokens": 768,
                "min_tokens": 100
            },
            "enable_progress": True,
            "batch_size": 3
        }
    }
