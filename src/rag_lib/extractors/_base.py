"""
Base extractor classes for information and entity extraction.

This module provides the abstract base classes for all extractors in the RAG system.
Extractors are responsible for identifying and extracting specific information
and entities from documents and queries.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from ..schemas.schema import Document, Query


class EntityType(Enum):
    """Enum for different types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"  
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    PHONE = "phone"
    EMAIL = "email"
    URL = "url"
    PRODUCT = "product"
    EVENT = "event"
    MISCELLANEOUS = "miscellaneous"


@dataclass
class ExtractedEntity:
    """
    Represents an extracted entity from text.
    
    Attributes:
        text (str): The original text of the entity
        entity_type (EntityType): The type of entity
        confidence (float): Confidence score (0.0 to 1.0)
        start_pos (int): Start position in the original text
        end_pos (int): End position in the original text
        metadata (Dict[str, Any]): Additional metadata about the entity
        normalized_value (Optional[str]): Normalized/standardized value of the entity
    """
    text: str
    entity_type: EntityType
    confidence: float = 1.0
    start_pos: int = -1
    end_pos: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)
    normalized_value: Optional[str] = None
    
    def __post_init__(self):
        """Validate extracted entity after initialization."""
        if not self.text.strip():
            raise ValueError("Entity text cannot be empty")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.start_pos >= 0 and self.end_pos >= 0 and self.start_pos >= self.end_pos:
            raise ValueError("start_pos must be less than end_pos")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "metadata": self.metadata,
            "normalized_value": self.normalized_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedEntity':
        """Create ExtractedEntity from dictionary representation."""
        data = data.copy()
        data["entity_type"] = EntityType(data["entity_type"])
        return cls(**data)


@dataclass
class ExtractedInformation:
    """
    Represents general extracted information from text.
    
    Attributes:
        content (str): The extracted information content
        confidence (float): Confidence score (0.0 to 1.0)
        source_text (Optional[str]): Original text that this information was extracted from
        metadata (Dict[str, Any]): Additional metadata about the information
    """
    content: str
    confidence: float = 1.0
    source_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate extracted information after initialization."""
        if not self.content.strip():
            raise ValueError("Information content cannot be empty")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert information to dictionary representation."""
        return {
            "content": self.content,
            "confidence": self.confidence,
            "source_text": self.source_text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedInformation':
        """Create ExtractedInformation from dictionary representation."""
        return cls(**data)


@dataclass
class ExtractionResult:
    """
    Represents the complete result of an extraction operation.
    
    Attributes:
        source_id (str): ID of the source document or query
        source_type (str): Type of source ("document" or "query")
        entities (List[ExtractedEntity]): List of extracted entities
        information (List[ExtractedInformation]): List of extracted information
        extraction_metadata (Dict[str, Any]): Metadata about the extraction process
        extraction_time (Optional[float]): Time taken for extraction in seconds
        extractor_name (str): Name of the extractor used
    """
    source_id: str
    source_type: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    information: List[ExtractedInformation] = field(default_factory=list)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_time: Optional[float] = None
    extractor_name: str = "base_extractor"
    
    def __post_init__(self):
        """Validate extraction result after initialization."""
        if not self.source_id.strip():
            raise ValueError("Source ID cannot be empty")
        
        if self.source_type not in ["document", "query", "text"]:
            raise ValueError("Source type must be 'document', 'query', or 'text'")
        
        if self.extraction_time is not None and self.extraction_time < 0:
            raise ValueError("Extraction time must be non-negative")
        
        # Set default metadata
        if "extracted_at" not in self.extraction_metadata:
            self.extraction_metadata["extracted_at"] = datetime.now().isoformat()
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[ExtractedEntity]:
        """Get all entities of a specific type."""
        return [entity for entity in self.entities if entity.entity_type == entity_type]
    
    def get_unique_entity_types(self) -> Set[EntityType]:
        """Get set of unique entity types found."""
        return {entity.entity_type for entity in self.entities}
    
    def get_total_extractions(self) -> int:
        """Get total number of extractions (entities + information)."""
        return len(self.entities) + len(self.information)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert extraction result to dictionary representation."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "entities": [entity.to_dict() for entity in self.entities],
            "information": [info.to_dict() for info in self.information],
            "extraction_metadata": self.extraction_metadata,
            "extraction_time": self.extraction_time,
            "extractor_name": self.extractor_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create ExtractionResult from dictionary representation."""
        data = data.copy()
        data["entities"] = [ExtractedEntity.from_dict(e) for e in data.get("entities", [])]
        data["information"] = [ExtractedInformation.from_dict(i) for i in data.get("information", [])]
        return cls(**data)


class ExtractorError(Exception):
    """Base exception for extractor operations."""
    pass


class ExtractionFailedError(ExtractorError):
    """Raised when extraction process fails."""
    pass


class BaseExtractor(ABC):
    """
    Abstract base class for all extractors.
    
    Extractors analyze text content and extract specific entities or information
    from documents and queries to enhance search and retrieval capabilities.
    
    Attributes:
        name (str): Name/identifier of the extractor
        config (Dict[str, Any]): Extractor-specific configuration parameters
        supported_entity_types (Set[EntityType]): Types of entities this extractor can extract
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        supported_entity_types: Optional[Set[EntityType]] = None
    ):
        """
        Initialize the extractor.
        
        Args:
            name (str): Name/identifier of the extractor
            config (Optional[Dict[str, Any]]): Extractor-specific configuration
            supported_entity_types (Optional[Set[EntityType]]): Supported entity types
        """
        self.name = name
        self.config = config or {}
        self.supported_entity_types = supported_entity_types or set()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate extractor configuration parameters."""
        # Default implementation does nothing
        # Concrete classes should override this for specific validation
        pass
    
    @abstractmethod
    def extract_from_document(self, document: Document) -> ExtractionResult:
        """
        Extract entities and information from a document.
        
        Args:
            document (Document): Document to extract from
            
        Returns:
            ExtractionResult: Extraction results
            
        Raises:
            ExtractorError: If extraction fails
            ValueError: If document is invalid
        """
        pass
    
    @abstractmethod
    def extract_from_query(self, query: Query) -> ExtractionResult:
        """
        Extract entities and information from a query.
        
        Args:
            query (Query): Query to extract from
            
        Returns:
            ExtractionResult: Extraction results
            
        Raises:
            ExtractorError: If extraction fails
            ValueError: If query is invalid
        """
        pass
    
    def extract_from_text(self, text: str, source_id: str = None) -> ExtractionResult:
        """
        Extract entities and information from raw text.
        
        Args:
            text (str): Text to extract from
            source_id (str): Optional ID for the source text
            
        Returns:
            ExtractionResult: Extraction results
        """
        if source_id is None:
            source_id = str(uuid.uuid4())
        
        # Create a temporary document for extraction
        temp_doc = Document(
            id=source_id,
            content=text,
            metadata={"temporary": True}
        )
        
        result = self.extract_from_document(temp_doc)
        result.source_type = "text"
        return result
    
    def supports_entity_type(self, entity_type: EntityType) -> bool:
        """Check if this extractor supports a specific entity type."""
        return entity_type in self.supported_entity_types
    
    def _create_extraction_result(
        self,
        source_id: str,
        source_type: str,
        entities: Optional[List[ExtractedEntity]] = None,
        information: Optional[List[ExtractedInformation]] = None,
        extraction_time: Optional[float] = None,
        **metadata
    ) -> ExtractionResult:
        """
        Helper method to create ExtractionResult with consistent metadata.
        """
        extraction_metadata = {
            "extractor_config": self.config,
            "supported_entity_types": [t.value for t in self.supported_entity_types],
            **metadata
        }
        
        return ExtractionResult(
            source_id=source_id,
            source_type=source_type,
            entities=entities or [],
            information=information or [],
            extraction_metadata=extraction_metadata,
            extraction_time=extraction_time,
            extractor_name=self.name
        )
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get information about the extractor."""
        return {
            "name": self.name,
            "config": self.config,
            "type": self.__class__.__name__,
            "supported_entity_types": [t.value for t in self.supported_entity_types]
        }
    
    def __str__(self) -> str:
        """String representation of the extractor."""
        return f"{self.__class__.__name__}(name='{self.name}')"


# Utility functions

def create_entity(
    text: str,
    entity_type: EntityType,
    confidence: float = 1.0,
    start_pos: int = -1,
    end_pos: int = -1,
    normalized_value: Optional[str] = None,
    **metadata
) -> ExtractedEntity:
    """Convenience function to create an ExtractedEntity."""
    return ExtractedEntity(
        text=text,
        entity_type=entity_type,
        confidence=confidence,
        start_pos=start_pos,
        end_pos=end_pos,
        normalized_value=normalized_value,
        metadata=metadata
    )


def create_information(
    content: str,
    confidence: float = 1.0,
    source_text: Optional[str] = None,
    **metadata
) -> ExtractedInformation:
    """Convenience function to create ExtractedInformation."""
    return ExtractedInformation(
        content=content,
        confidence=confidence,
        source_text=source_text,
        metadata=metadata
    )