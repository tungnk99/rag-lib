"""
LLM-based extractors for flexible information and entity extraction.

This module provides extractors that use Large Language Models to extract
information and entities from text using customizable prompts.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Union
import json
import re
import time
from dataclasses import dataclass, field

from ._base import (
    BaseExtractor, ExtractorError, ExtractionResult,
    ExtractedEntity, ExtractedInformation, EntityType,
    create_entity, create_information
)
from ..schemas.schema import Document, Query


@dataclass
class ExtractionPrompt:
    """
    Represents a prompt template for LLM-based extraction.
    
    Attributes:
        name (str): Name/identifier of the prompt
        template (str): The prompt template with placeholders
        description (str): Description of what this prompt extracts
        expected_format (str): Expected output format description
        examples (List[str]): Example inputs/outputs for few-shot learning
        temperature (float): LLM temperature for this prompt
        max_tokens (int): Maximum tokens for LLM response
    """
    name: str
    template: str
    description: str = ""
    expected_format: str = "JSON"
    examples: List[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 1000
    
    def format_prompt(self, text: str, **kwargs) -> str:
        """
        Format the prompt template with given text and additional arguments.
        
        Args:
            text (str): Text content to extract from
            **kwargs: Additional variables for prompt formatting
            
        Returns:
            str: Formatted prompt
        """
        # Add examples if provided
        examples_text = ""
        if self.examples:
            examples_text = "\n\nExamples:\n" + "\n".join(self.examples)
        
        # Format the main template
        formatted_prompt = self.template.format(
            text=text,
            examples=examples_text,
            **kwargs
        )
        
        return formatted_prompt


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This allows the extractor to work with different LLM services
    (OpenAI, Anthropic, local models, etc.)
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt (str): Input prompt
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            str: Generated text
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the LLM provider."""
        pass


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing and development.
    """
    
    def __init__(self, responses: Optional[List[str]] = None):
        """
        Initialize mock provider.
        
        Args:
            responses (Optional[List[str]]): Predefined responses to cycle through
        """
        self.responses = responses or [
            '{"entities": [], "information": ["Mock extraction result"]}'
        ]
        self.call_count = 0
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        time.sleep(0.1)  # Simulate API delay
        return response
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get mock provider info."""
        return {
            "provider": "mock",
            "model": "mock-model",
            "responses_count": len(self.responses),
            "calls_made": self.call_count
        }


class LLMExtractor(BaseExtractor):
    """
    LLM-based extractor that uses customizable prompts for extraction.
    
    This extractor allows flexible information extraction by using
    different prompts for different extraction tasks.
    """
    
    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        extraction_prompts: Optional[Dict[str, ExtractionPrompt]] = None,
        default_prompt_name: str = "general",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize the LLM extractor.
        
        Args:
            name (str): Name of the extractor
            llm_provider (LLMProvider): LLM provider to use
            extraction_prompts (Optional[Dict[str, ExtractionPrompt]]): Custom prompts
            default_prompt_name (str): Name of default prompt to use
            max_retries (int): Maximum number of retries on failure
            retry_delay (float): Delay between retries in seconds
            **kwargs: Additional configuration
        """
        super().__init__(name=name, config=kwargs)
        self.llm_provider = llm_provider
        self.extraction_prompts = extraction_prompts or self._get_default_prompts()
        self.default_prompt_name = default_prompt_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set supported entity types based on default prompts
        self.supported_entity_types = {
            EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION,
            EntityType.DATE, EntityType.TIME, EntityType.MONEY,
            EntityType.EMAIL, EntityType.PHONE, EntityType.URL,
            EntityType.PRODUCT, EntityType.EVENT, EntityType.MISCELLANEOUS
        }
    
    def _get_default_prompts(self) -> Dict[str, ExtractionPrompt]:
        """Get default extraction prompts."""
        return {
            "general": ExtractionPrompt(
                name="general",
                template="""Extract useful entities and information from the following text.

Text: {text}

Please extract:
1. Entities (people, organizations, locations, dates, etc.)
2. Key information or insights

Return your response in JSON format:
{{
  "entities": [
    {{
      "text": "entity text",
      "type": "entity_type",
      "confidence": 0.95
    }}
  ],
  "information": [
    {{
      "content": "extracted information",
      "confidence": 0.9
    }}
  ]
}}

Only return the JSON response, no additional text.{examples}""",
                description="General purpose extraction",
                temperature=0.0,
                max_tokens=1000
            ),
            
            "entities_only": ExtractionPrompt(
                name="entities_only",
                template="""Extract named entities from the following text.

Text: {text}

Extract entities such as:
- People (PERSON)
- Organizations (ORGANIZATION)
- Locations (LOCATION)
- Dates (DATE)
- Times (TIME)
- Money amounts (MONEY)
- Email addresses (EMAIL)
- Phone numbers (PHONE)
- URLs (URL)
- Products (PRODUCT)
- Events (EVENT)

Return JSON format:
{{
  "entities": [
    {{
      "text": "entity text",
      "type": "entity_type",
      "confidence": 0.95,
      "start_pos": 10,
      "end_pos": 20
    }}
  ]
}}

Only return the JSON response.{examples}""",
                description="Entity extraction only",
                temperature=0.0,
                max_tokens=800
            ),
            
            "information_only": ExtractionPrompt(
                name="information_only",
                template="""Extract key information and insights from the following text.

Text: {text}

Extract important information such as:
- Key facts
- Main topics
- Summary points
- Important insights
- Conclusions

Return JSON format:
{{
  "information": [
    {{
      "content": "extracted information",
      "confidence": 0.9,
      "metadata": {{
        "type": "fact/topic/summary/insight"
      }}
    }}
  ]
}}

Only return the JSON response.{examples}""",
                description="Information extraction only",
                temperature=0.1,
                max_tokens=1000
            ),
            
            "query_analysis": ExtractionPrompt(
                name="query_analysis",
                template="""Analyze the following query and extract relevant information.

Query: {text}

Extract:
- Intent of the query
- Key entities mentioned
- Important concepts
- Context information

Return JSON format:
{{
  "entities": [
    {{
      "text": "entity text",
      "type": "entity_type",
      "confidence": 0.95
    }}
  ],
  "information": [
    {{
      "content": "query intent or analysis",
      "confidence": 0.9,
      "metadata": {{
        "type": "intent/concept/context"
      }}
    }}
  ]
}}

Only return the JSON response.{examples}""",
                description="Query analysis and extraction",
                temperature=0.0,
                max_tokens=800
            )
        }
    
    def add_prompt(self, prompt: ExtractionPrompt) -> None:
        """Add a new extraction prompt."""
        self.extraction_prompts[prompt.name] = prompt
    
    def get_prompt(self, name: str) -> Optional[ExtractionPrompt]:
        """Get an extraction prompt by name."""
        return self.extraction_prompts.get(name)
    
    def list_prompts(self) -> List[str]:
        """List available prompt names."""
        return list(self.extraction_prompts.keys())
    
    def extract_from_document(
        self,
        document: Document,
        prompt_name: Optional[str] = None,
        **prompt_kwargs
    ) -> ExtractionResult:
        """
        Extract entities and information from a document using LLM.
        
        Args:
            document (Document): Document to extract from
            prompt_name (Optional[str]): Name of prompt to use
            **prompt_kwargs: Additional variables for prompt formatting
            
        Returns:
            ExtractionResult: Extraction results
        """
        start_time = time.time()
        
        try:
            # Use specified prompt or default
            prompt_name = prompt_name or self.default_prompt_name
            prompt = self.extraction_prompts.get(prompt_name)
            
            if not prompt:
                raise ExtractorError(f"Prompt '{prompt_name}' not found")
            
            # Extract using the prompt
            entities, information = self._extract_with_prompt(
                document.content, prompt, **prompt_kwargs
            )
            
            # Calculate extraction time
            extraction_time = time.time() - start_time
            
            # Create result
            return self._create_extraction_result(
                source_id=document.id,
                source_type="document",
                entities=entities,
                information=information,
                extraction_time=extraction_time,
                prompt_used=prompt_name,
                llm_provider=self.llm_provider.get_provider_info()
            )
            
        except Exception as e:
            raise ExtractorError(f"Failed to extract from document {document.id}: {str(e)}")
    
    def extract_from_query(
        self,
        query: Query,
        prompt_name: Optional[str] = None,
        **prompt_kwargs
    ) -> ExtractionResult:
        """
        Extract entities and information from a query using LLM.
        
        Args:
            query (Query): Query to extract from
            prompt_name (Optional[str]): Name of prompt to use (defaults to 'query_analysis')
            **prompt_kwargs: Additional variables for prompt formatting
            
        Returns:
            ExtractionResult: Extraction results
        """
        start_time = time.time()
        
        try:
            # Use query-specific prompt if not specified
            prompt_name = prompt_name or "query_analysis"
            prompt = self.extraction_prompts.get(prompt_name)
            
            if not prompt:
                raise ExtractorError(f"Prompt '{prompt_name}' not found")
            
            # Extract using the prompt
            entities, information = self._extract_with_prompt(
                query.content, prompt, **prompt_kwargs
            )
            
            # Calculate extraction time
            extraction_time = time.time() - start_time
            
            # Create result
            query_id = query.metadata.get('query_id', 'unknown')
            return self._create_extraction_result(
                source_id=query_id,
                source_type="query",
                entities=entities,
                information=information,
                extraction_time=extraction_time,
                prompt_used=prompt_name,
                llm_provider=self.llm_provider.get_provider_info()
            )
            
        except Exception as e:
            query_id = query.metadata.get('query_id', 'unknown')
            raise ExtractorError(f"Failed to extract from query {query_id}: {str(e)}")
    
    def _extract_with_prompt(
        self,
        text: str,
        prompt: ExtractionPrompt,
        **prompt_kwargs
    ) -> tuple[List[ExtractedEntity], List[ExtractedInformation]]:
        """
        Perform extraction using a specific prompt.
        
        Args:
            text (str): Text to extract from
            prompt (ExtractionPrompt): Prompt to use
            **prompt_kwargs: Additional prompt variables
            
        Returns:
            tuple: (entities, information)
        """
        # Format the prompt
        formatted_prompt = prompt.format_prompt(text, **prompt_kwargs)
        
        # Call LLM with retries
        response = self._call_llm_with_retries(
            formatted_prompt,
            prompt.temperature,
            prompt.max_tokens
        )
        
        # Parse the response
        entities, information = self._parse_llm_response(response, text)
        
        return entities, information
    
    def _call_llm_with_retries(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Call LLM with retry logic.
        
        Args:
            prompt (str): Formatted prompt
            temperature (float): LLM temperature
            max_tokens (int): Maximum tokens
            
        Returns:
            str: LLM response
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_provider.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
        raise ExtractorError(f"LLM call failed after {self.max_retries} attempts: {str(last_error)}")
    
    def _parse_llm_response(
        self,
        response: str,
        original_text: str
    ) -> tuple[List[ExtractedEntity], List[ExtractedInformation]]:
        """
        Parse LLM response to extract entities and information.
        
        Args:
            response (str): LLM response
            original_text (str): Original text for position finding
            
        Returns:
            tuple: (entities, information)
        """
        entities = []
        information = []
        
        try:
            # Try to extract JSON from response
            json_response = self._extract_json_from_response(response)
            
            # Parse entities
            if "entities" in json_response:
                for entity_data in json_response["entities"]:
                    try:
                        entity = self._create_entity_from_json(entity_data, original_text)
                        entities.append(entity)
                    except Exception as e:
                        print(f"Warning: Failed to create entity from {entity_data}: {e}")
            
            # Parse information
            if "information" in json_response:
                for info_data in json_response["information"]:
                    try:
                        info = self._create_information_from_json(info_data, original_text)
                        information.append(info)
                    except Exception as e:
                        print(f"Warning: Failed to create information from {info_data}: {e}")
            
        except Exception as e:
            # If JSON parsing fails, try to extract as plain text
            print(f"Warning: JSON parsing failed, using fallback: {e}")
            fallback_info = create_information(
                content=response.strip(),
                confidence=0.5,
                source_text=original_text[:100] + "..." if len(original_text) > 100 else original_text,
                metadata={"extraction_method": "fallback", "raw_response": True}
            )
            information.append(fallback_info)
        
        return entities, information
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON object from LLM response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # If no JSON found, try parsing the entire response
            return json.loads(response.strip())
    
    def _create_entity_from_json(
        self,
        entity_data: Dict[str, Any],
        original_text: str
    ) -> ExtractedEntity:
        """Create ExtractedEntity from JSON data."""
        text = entity_data.get("text", "")
        entity_type_str = entity_data.get("type", "miscellaneous").lower()
        
        # Map entity type string to enum
        entity_type = self._map_entity_type(entity_type_str)
        
        confidence = float(entity_data.get("confidence", 1.0))
        start_pos = entity_data.get("start_pos", -1)
        end_pos = entity_data.get("end_pos", -1)
        
        # Try to find positions if not provided
        if start_pos == -1 and text in original_text:
            start_pos = original_text.find(text)
            end_pos = start_pos + len(text)
        
        metadata = entity_data.get("metadata", {})
        normalized_value = entity_data.get("normalized_value")
        
        return create_entity(
            text=text,
            entity_type=entity_type,
            confidence=confidence,
            start_pos=start_pos,
            end_pos=end_pos,
            normalized_value=normalized_value,
            **metadata
        )
    
    def _create_information_from_json(
        self,
        info_data: Dict[str, Any],
        original_text: str
    ) -> ExtractedInformation:
        """Create ExtractedInformation from JSON data."""
        content = info_data.get("content", "")
        confidence = float(info_data.get("confidence", 1.0))
        source_text = info_data.get("source_text")
        metadata = info_data.get("metadata", {})
        
        return create_information(
            content=content,
            confidence=confidence,
            source_text=source_text,
            **metadata
        )
    
    def _map_entity_type(self, entity_type_str: str) -> EntityType:
        """Map string to EntityType enum."""
        type_mapping = {
            "person": EntityType.PERSON,
            "organization": EntityType.ORGANIZATION,
            "location": EntityType.LOCATION,
            "date": EntityType.DATE,
            "time": EntityType.TIME,
            "money": EntityType.MONEY,
            "phone": EntityType.PHONE,
            "email": EntityType.EMAIL,
            "url": EntityType.URL,
            "product": EntityType.PRODUCT,
            "event": EntityType.EVENT,
            "miscellaneous": EntityType.MISCELLANEOUS
        }
        
        return type_mapping.get(entity_type_str.lower(), EntityType.MISCELLANEOUS)
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get detailed information about the extractor."""
        info = super().get_extractor_info()
        info.update({
            "llm_provider": self.llm_provider.get_provider_info(),
            "available_prompts": list(self.extraction_prompts.keys()),
            "default_prompt": self.default_prompt_name,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay
        })
        return info


# Convenience function for creating LLM extractors

def create_llm_extractor(
    name: str,
    llm_provider: LLMProvider,
    custom_prompts: Optional[Dict[str, ExtractionPrompt]] = None,
    default_prompt: str = "general",
    **kwargs
) -> LLMExtractor:
    """
    Convenience function to create an LLM extractor.
    
    Args:
        name (str): Name of the extractor
        llm_provider (LLMProvider): LLM provider to use
        custom_prompts (Optional[Dict[str, ExtractionPrompt]]): Custom prompts
        default_prompt (str): Default prompt name
        **kwargs: Additional configuration
        
    Returns:
        LLMExtractor: Configured LLM extractor
    """
    return LLMExtractor(
        name=name,
        llm_provider=llm_provider,
        extraction_prompts=custom_prompts,
        default_prompt_name=default_prompt,
        **kwargs
    )
