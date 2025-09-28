"""
Reader factory for automatic reader selection and management.

This module provides factory classes for creating and managing readers
for different file formats, with automatic format detection and reader selection.
"""

from typing import Dict, List, Optional, Type, Any, Set, Callable
from pathlib import Path
import mimetypes
import logging
from dataclasses import dataclass, field

from ._base import BaseReader
from ..schemas.schema import FileContent


@dataclass
class ReaderConfig:
    """
    Configuration for a specific reader.
    
    Attributes:
        reader_class (Type[BaseReader]): The reader class
        extensions (Set[str]): Supported file extensions (without dots)
        mime_types (Set[str]): Supported MIME types
        priority (int): Priority for selection (higher = preferred)
        kwargs (Dict[str, Any]): Default arguments for reader initialization
        enabled (bool): Whether this reader is enabled
        description (str): Description of the reader
    """
    reader_class: Type[BaseReader]
    extensions: Set[str] = field(default_factory=set)
    mime_types: Set[str] = field(default_factory=set)
    priority: int = 0
    kwargs: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    
    def create_reader(self, **override_kwargs) -> BaseReader:
        """
        Create an instance of the reader.
        
        Args:
            **override_kwargs: Arguments to override default kwargs
            
        Returns:
            BaseReader: Created reader instance
        """
        final_kwargs = {**self.kwargs, **override_kwargs}
        return self.reader_class(**final_kwargs)
    
    def supports_extension(self, extension: str) -> bool:
        """Check if this reader supports the given extension."""
        return extension.lower().lstrip('.') in self.extensions
    
    def supports_mime_type(self, mime_type: str) -> bool:
        """Check if this reader supports the given MIME type."""
        return mime_type.lower() in self.mime_types


class ReaderRegistryError(Exception):
    """Exception raised by reader registry operations."""
    pass


class ReaderFactory:
    """
    Factory for creating and managing document readers.
    
    Automatically selects appropriate readers based on file format,
    extension, and MIME type. Supports reader registration, configuration,
    and fallback mechanisms.
    """
    
    def __init__(self, auto_register: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialize the reader factory.
        
        Args:
            auto_register (bool): Whether to automatically register available readers
            logger (Optional[logging.Logger]): Logger for factory operations
        """
        self.logger = logger or self._setup_default_logger()
        self._registry: Dict[str, ReaderConfig] = {}
        self._extension_map: Dict[str, List[str]] = {}  # extension -> reader names
        self._mime_type_map: Dict[str, List[str]] = {}  # mime_type -> reader names
        
        if auto_register:
            self._auto_register_readers()
    
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger for the factory."""
        logger = logging.getLogger("ReaderFactory")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)  # Less verbose by default
        return logger
    
    def _auto_register_readers(self):
        """Automatically register available readers."""
        self.logger.info("Auto-registering available readers...")
        
        # Register TXT reader (always available)
        try:
            from .txt_reader import TxtReader
            self.register_reader(
                "txt",
                TxtReader,
                extensions={"txt", "text"},
                mime_types={"text/plain"},
                priority=1,
                description="Plain text file reader"
            )
        except ImportError as e:
            self.logger.warning(f"Could not register TXT reader: {e}")
        
        # Register PDF reader
        try:
            from .pdf_reader import PdfReader
            self.register_reader(
                "pdf",
                PdfReader,
                extensions={"pdf"},
                mime_types={"application/pdf"},
                priority=5,
                description="PDF document reader"
            )
        except ImportError as e:
            self.logger.debug(f"PDF reader not available: {e}")
        
        # Register DOCX reader
        try:
            from .docx_reader import DocxReader
            self.register_reader(
                "docx",
                DocxReader,
                extensions={"docx"},
                mime_types={"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
                priority=5,
                description="Microsoft Word document reader"
            )
        except ImportError as e:
            self.logger.debug(f"DOCX reader not available: {e}")
        
        # Register XLSX reader
        try:
            from .xlsx_reader import XlsxReader
            self.register_reader(
                "xlsx",
                XlsxReader,
                extensions={"xlsx", "xlsm"},
                mime_types={
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.ms-excel.sheet.macroEnabled.12"
                },
                priority=5,
                description="Microsoft Excel spreadsheet reader"
            )
        except ImportError as e:
            self.logger.debug(f"XLSX reader not available: {e}")
        
        self.logger.info(f"Registered {len(self._registry)} readers")
    
    def register_reader(
        self,
        name: str,
        reader_class: Type[BaseReader],
        extensions: Optional[Set[str]] = None,
        mime_types: Optional[Set[str]] = None,
        priority: int = 0,
        enabled: bool = True,
        description: str = "",
        **kwargs
    ) -> None:
        """
        Register a reader for specific file formats.
        
        Args:
            name (str): Unique name for the reader
            reader_class (Type[BaseReader]): Reader class to register
            extensions (Optional[Set[str]]): Supported file extensions
            mime_types (Optional[Set[str]]): Supported MIME types
            priority (int): Priority for selection (higher = preferred)
            enabled (bool): Whether reader is enabled
            description (str): Description of the reader
            **kwargs: Default arguments for reader initialization
        """
        if name in self._registry:
            self.logger.warning(f"Reader '{name}' already registered, overwriting")
        
        # Normalize extensions (remove dots, convert to lowercase)
        normalized_extensions = set()
        if extensions:
            for ext in extensions:
                normalized_extensions.add(ext.lower().lstrip('.'))
        
        # Normalize MIME types
        normalized_mime_types = set()
        if mime_types:
            for mime_type in mime_types:
                normalized_mime_types.add(mime_type.lower())
        
        # Create config
        config = ReaderConfig(
            reader_class=reader_class,
            extensions=normalized_extensions,
            mime_types=normalized_mime_types,
            priority=priority,
            kwargs=kwargs,
            enabled=enabled,
            description=description
        )
        
        # Register
        self._registry[name] = config
        
        # Update extension mapping
        for ext in normalized_extensions:
            if ext not in self._extension_map:
                self._extension_map[ext] = []
            if name not in self._extension_map[ext]:
                self._extension_map[ext].append(name)
                # Sort by priority (descending)
                self._extension_map[ext].sort(
                    key=lambda x: self._registry[x].priority,
                    reverse=True
                )
        
        # Update MIME type mapping
        for mime_type in normalized_mime_types:
            if mime_type not in self._mime_type_map:
                self._mime_type_map[mime_type] = []
            if name not in self._mime_type_map[mime_type]:
                self._mime_type_map[mime_type].append(name)
                # Sort by priority (descending)
                self._mime_type_map[mime_type].sort(
                    key=lambda x: self._registry[x].priority,
                    reverse=True
                )
        
        self.logger.debug(f"Registered reader '{name}' for extensions {normalized_extensions}")
    
    def unregister_reader(self, name: str) -> None:
        """
        Unregister a reader.
        
        Args:
            name (str): Name of reader to unregister
        """
        if name not in self._registry:
            raise ReaderRegistryError(f"Reader '{name}' not registered")
        
        config = self._registry[name]
        
        # Remove from extension mapping
        for ext in config.extensions:
            if ext in self._extension_map and name in self._extension_map[ext]:
                self._extension_map[ext].remove(name)
                if not self._extension_map[ext]:
                    del self._extension_map[ext]
        
        # Remove from MIME type mapping
        for mime_type in config.mime_types:
            if mime_type in self._mime_type_map and name in self._mime_type_map[mime_type]:
                self._mime_type_map[mime_type].remove(name)
                if not self._mime_type_map[mime_type]:
                    del self._mime_type_map[mime_type]
        
        # Remove from registry
        del self._registry[name]
        self.logger.debug(f"Unregistered reader '{name}'")
    
    def get_reader_for_file(
        self,
        file_path: str,
        preferred_reader: Optional[str] = None,
        **reader_kwargs
    ) -> BaseReader:
        """
        Get the best reader for a specific file.
        
        Args:
            file_path (str): Path to the file
            preferred_reader (Optional[str]): Preferred reader name
            **reader_kwargs: Arguments to pass to reader initialization
            
        Returns:
            BaseReader: Appropriate reader instance
            
        Raises:
            ReaderRegistryError: If no suitable reader found
        """
        # If preferred reader specified, try it first
        if preferred_reader:
            if preferred_reader in self._registry:
                config = self._registry[preferred_reader]
                if config.enabled:
                    reader = config.create_reader(**reader_kwargs)
                    if reader.supports_format(file_path):
                        self.logger.debug(f"Using preferred reader '{preferred_reader}' for {file_path}")
                        return reader
                    else:
                        self.logger.warning(
                            f"Preferred reader '{preferred_reader}' does not support {file_path}"
                        )
            else:
                self.logger.warning(f"Preferred reader '{preferred_reader}' not registered")
        
        # Try to find reader by extension
        path = Path(file_path)
        extension = path.suffix.lower().lstrip('.')
        
        if extension and extension in self._extension_map:
            for reader_name in self._extension_map[extension]:
                config = self._registry[reader_name]
                if config.enabled:
                    reader = config.create_reader(**reader_kwargs)
                    if reader.supports_format(file_path):
                        self.logger.debug(f"Selected reader '{reader_name}' for extension '{extension}'")
                        return reader
        
        # Try to find reader by MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            mime_type = mime_type.lower()
            if mime_type in self._mime_type_map:
                for reader_name in self._mime_type_map[mime_type]:
                    config = self._registry[reader_name]
                    if config.enabled:
                        reader = config.create_reader(**reader_kwargs)
                        if reader.supports_format(file_path):
                            self.logger.debug(f"Selected reader '{reader_name}' for MIME type '{mime_type}'")
                            return reader
        
        # Try all enabled readers as fallback
        self.logger.debug(f"Trying fallback readers for {file_path}")
        for reader_name, config in sorted(
            self._registry.items(),
            key=lambda x: x[1].priority,
            reverse=True
        ):
            if config.enabled:
                reader = config.create_reader(**reader_kwargs)
                if reader.supports_format(file_path):
                    self.logger.debug(f"Fallback reader '{reader_name}' supports {file_path}")
                    return reader
        
        # No reader found
        available_extensions = set()
        for config in self._registry.values():
            if config.enabled:
                available_extensions.update(config.extensions)
        
        raise ReaderRegistryError(
            f"No suitable reader found for file: {file_path}\n"
            f"Available extensions: {sorted(available_extensions)}"
        )
    
    def read_file(
        self,
        file_path: str,
        preferred_reader: Optional[str] = None,
        **reader_kwargs
    ) -> FileContent:
        """
        Read a file using the most appropriate reader.
        
        Args:
            file_path (str): Path to the file to read
            preferred_reader (Optional[str]): Preferred reader name
            **reader_kwargs: Arguments to pass to reader
            
        Returns:
            FileContent: File content object
        """
        reader = self.get_reader_for_file(file_path, preferred_reader, **reader_kwargs)
        return reader.read(file_path)
    
    def get_supported_extensions(self) -> Set[str]:
        """Get all supported file extensions."""
        extensions = set()
        for config in self._registry.values():
            if config.enabled:
                extensions.update(config.extensions)
        return extensions
    
    def get_supported_mime_types(self) -> Set[str]:
        """Get all supported MIME types."""
        mime_types = set()
        for config in self._registry.values():
            if config.enabled:
                mime_types.update(config.mime_types)
        return mime_types
    
    def list_readers(self, enabled_only: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        List all registered readers with their information.
        
        Args:
            enabled_only (bool): Whether to include only enabled readers
            
        Returns:
            Dict[str, Dict[str, Any]]: Reader information
        """
        readers_info = {}
        for name, config in self._registry.items():
            if not enabled_only or config.enabled:
                readers_info[name] = {
                    "class": config.reader_class.__name__,
                    "extensions": sorted(config.extensions),
                    "mime_types": sorted(config.mime_types),
                    "priority": config.priority,
                    "enabled": config.enabled,
                    "description": config.description,
                    "kwargs": config.kwargs
                }
        return readers_info
    
    def enable_reader(self, name: str) -> None:
        """Enable a registered reader."""
        if name not in self._registry:
            raise ReaderRegistryError(f"Reader '{name}' not registered")
        self._registry[name].enabled = True
        self.logger.debug(f"Enabled reader '{name}'")
    
    def disable_reader(self, name: str) -> None:
        """Disable a registered reader."""
        if name not in self._registry:
            raise ReaderRegistryError(f"Reader '{name}' not registered")
        self._registry[name].enabled = False
        self.logger.debug(f"Disabled reader '{name}'")
    
    def update_reader_config(self, name: str, **kwargs) -> None:
        """
        Update configuration for a registered reader.
        
        Args:
            name (str): Reader name
            **kwargs: Configuration updates
        """
        if name not in self._registry:
            raise ReaderRegistryError(f"Reader '{name}' not registered")
        
        config = self._registry[name]
        config.kwargs.update(kwargs)
        self.logger.debug(f"Updated config for reader '{name}': {kwargs}")
    
    def supports_file(self, file_path: str) -> bool:
        """
        Check if any registered reader supports the given file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if file is supported
        """
        try:
            self.get_reader_for_file(file_path)
            return True
        except ReaderRegistryError:
            return False
    
    def get_factory_info(self) -> Dict[str, Any]:
        """Get information about the factory state."""
        return {
            "total_readers": len(self._registry),
            "enabled_readers": len([c for c in self._registry.values() if c.enabled]),
            "supported_extensions": sorted(self.get_supported_extensions()),
            "supported_mime_types": sorted(self.get_supported_mime_types()),
            "extension_mapping": {
                ext: [name for name in readers if self._registry[name].enabled]
                for ext, readers in self._extension_map.items()
            }
        }


# Global factory instance
_default_factory: Optional[ReaderFactory] = None


def get_default_factory() -> ReaderFactory:
    """Get or create the default reader factory."""
    global _default_factory
    if _default_factory is None:
        _default_factory = ReaderFactory(auto_register=True)
    return _default_factory


def read_file(
    file_path: str,
    preferred_reader: Optional[str] = None,
    factory: Optional[ReaderFactory] = None,
    **reader_kwargs
) -> FileContent:
    """
    Convenience function to read a file using the default factory.
    
    Args:
        file_path (str): Path to the file to read
        preferred_reader (Optional[str]): Preferred reader name
        factory (Optional[ReaderFactory]): Custom factory to use
        **reader_kwargs: Arguments to pass to reader
        
    Returns:
        FileContent: File content object
    """
    if factory is None:
        factory = get_default_factory()
    
    return factory.read_file(file_path, preferred_reader, **reader_kwargs)


def create_reader_factory(auto_register: bool = True, **kwargs) -> ReaderFactory:
    """
    Create a new reader factory.
    
    Args:
        auto_register (bool): Whether to auto-register available readers
        **kwargs: Additional factory arguments
        
    Returns:
        ReaderFactory: New factory instance
    """
    return ReaderFactory(auto_register=auto_register, **kwargs)


def get_supported_formats(factory: Optional[ReaderFactory] = None) -> Dict[str, List[str]]:
    """
    Get supported file formats.
    
    Args:
        factory (Optional[ReaderFactory]): Factory to use (default if None)
        
    Returns:
        Dict[str, List[str]]: Mapping of extensions to reader names
    """
    if factory is None:
        factory = get_default_factory()
    
    info = factory.get_factory_info()
    return info["extension_mapping"]
