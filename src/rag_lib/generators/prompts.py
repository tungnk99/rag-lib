"""
Prompt templates for text generation in RAG systems.

This module contains pre-built prompt templates and utilities for creating
custom prompts for various text generation tasks in RAG pipelines.
"""

from typing import List, Optional, Dict, Any
import re


class PromptTemplate:
    """
    Template class for building prompts with placeholders.
    
    This class allows creating reusable prompt templates with variable substitution,
    validation of required variables, and automatic variable extraction.
    """
    
    def __init__(self, template: str, required_variables: Optional[List[str]] = None):
        """
        Initialize prompt template.
        
        Args:
            template (str): Template string with placeholders using {variable} syntax
            required_variables (Optional[List[str]]): List of required variable names
        """
        self.template = template
        self.required_variables = required_variables or []
        self._validate_template()
    
    def _validate_template(self):
        """Validate the template format and required variables."""
        template_vars = self.get_variables()
        
        # Check if all required variables are in template
        missing_in_template = [var for var in self.required_variables if var not in template_vars]
        if missing_in_template:
            raise ValueError(f"Required variables not found in template: {missing_in_template}")
    
    def format(self, **kwargs) -> str:
        """
        Format template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in template
            
        Returns:
            str: Formatted prompt string
            
        Raises:
            ValueError: If required variables are missing or template formatting fails
        """
        # Check for required variables
        missing_vars = [var for var in self.required_variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template variable not provided: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting template: {e}")
    
    def get_variables(self) -> List[str]:
        """
        Extract variable names from template.
        
        Returns:
            List[str]: List of unique variable names found in template
        """
        return list(set(re.findall(r'\{(\w+)\}', self.template)))
    
    def validate_variables(self, **kwargs) -> bool:
        """
        Validate that all required variables are provided.
        
        Args:
            **kwargs: Variables to validate
            
        Returns:
            bool: True if all required variables are present
        """
        missing_vars = [var for var in self.required_variables if var not in kwargs]
        return len(missing_vars) == 0
    
    def get_sample_variables(self) -> Dict[str, str]:
        """
        Get sample variables for testing the template.
        
        Returns:
            Dict[str, str]: Dictionary with sample values for all template variables
        """
        variables = self.get_variables()
        sample_data = {
            'query': 'What is machine learning?',
            'context': 'Machine learning is a subset of artificial intelligence...',
            'question': 'What is machine learning?',
            'documents': 'Document 1: ML Overview\nDocument 2: ML Applications',
            'domain': 'Technology',
            'max_words': '200',
            'user': 'User',
            'assistant': 'Assistant',
            'system': 'System',
            'conversation_history': 'User: Hello\nAssistant: Hi there!'
        }
        
        return {var: sample_data.get(var, f'[{var.upper()}]') for var in variables}
    
    def preview(self) -> str:
        """
        Generate a preview of the template with sample data.
        
        Returns:
            str: Template formatted with sample variables
        """
        sample_vars = self.get_sample_variables()
        try:
            return self.format(**sample_vars)
        except Exception as e:
            return f"Error generating preview: {e}"
    
    def __str__(self) -> str:
        """String representation showing template info."""
        return f"PromptTemplate(variables={self.get_variables()}, required={self.required_variables})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"PromptTemplate(template='{self.template[:50]}...', required_variables={self.required_variables})"


# Pre-built prompt templates for common RAG tasks

DEFAULT_QA_TEMPLATE = PromptTemplate("""Based on the following context documents, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer: """, 
required_variables=['context', 'query'])

SUMMARIZATION_TEMPLATE = PromptTemplate("""Please provide a summary of the following documents that addresses this query: {query}

Documents:
{context}

Summary: """, 
required_variables=['context', 'query'])

CONVERSATIONAL_TEMPLATE = PromptTemplate("""You are a helpful assistant. Use the following context to answer the user's question in a conversational manner.

Context:
{context}

User: {query}

Assistant: """, 
required_variables=['context', 'query'])

ANALYTICAL_TEMPLATE = PromptTemplate("""Analyze the following documents and provide a detailed response to the question.

Documents:
{context}

Question: {query}

Please provide:
1. Key findings from the documents
2. Analysis and insights
3. Direct answer to the question

Analysis: """, 
required_variables=['context', 'query'])

COMPARISON_TEMPLATE = PromptTemplate("""Compare and contrast the information in the following documents regarding: {query}

Documents:
{context}

Comparison:
- Similarities:
- Differences:
- Conclusion:
""", 
required_variables=['context', 'query'])

FACTUAL_TEMPLATE = PromptTemplate("""Based strictly on the information provided in the documents below, answer the following question. If the information is not available in the documents, state that clearly.

Documents:
{context}

Question: {query}

Factual Answer: """, 
required_variables=['context', 'query'])

CITATION_TEMPLATE = PromptTemplate("""Answer the following question using the provided documents. Include citations to specific documents in your response.

Documents:
{context}

Question: {query}

Answer with citations: """, 
required_variables=['context', 'query'])

STEP_BY_STEP_TEMPLATE = PromptTemplate("""Provide a step-by-step explanation to answer the following question using the provided context.

Context:
{context}

Question: {query}

Step-by-step explanation:
1. """, 
required_variables=['context', 'query'])

MULTI_TURN_TEMPLATE = PromptTemplate("""You are having a conversation with a user. Use the context to provide helpful responses.

Context:
{context}

Conversation History:
{conversation_history}

User: {query}

Assistant: """, 
required_variables=['context', 'conversation_history', 'query'])


# Template collections for different use cases
QUESTION_ANSWERING_TEMPLATES = {
    'default': DEFAULT_QA_TEMPLATE,
    'factual': FACTUAL_TEMPLATE,
    'analytical': ANALYTICAL_TEMPLATE,
    'citation': CITATION_TEMPLATE,
    'step_by_step': STEP_BY_STEP_TEMPLATE
}

CONVERSATIONAL_TEMPLATES = {
    'default': CONVERSATIONAL_TEMPLATE,
    'multi_turn': MULTI_TURN_TEMPLATE
}

ANALYSIS_TEMPLATES = {
    'comparison': COMPARISON_TEMPLATE,
    'analytical': ANALYTICAL_TEMPLATE,
    'summarization': SUMMARIZATION_TEMPLATE
}

ALL_TEMPLATES = {
    'qa_default': DEFAULT_QA_TEMPLATE,
    'summarization': SUMMARIZATION_TEMPLATE,
    'conversational': CONVERSATIONAL_TEMPLATE,
    'analytical': ANALYTICAL_TEMPLATE,
    'comparison': COMPARISON_TEMPLATE,
    'factual': FACTUAL_TEMPLATE,
    'citation': CITATION_TEMPLATE,
    'step_by_step': STEP_BY_STEP_TEMPLATE,
    'multi_turn': MULTI_TURN_TEMPLATE
}


# Utility functions
def get_template(template_name: str) -> PromptTemplate:
    """
    Get a template by name.
    
    Args:
        template_name (str): Name of the template
        
    Returns:
        PromptTemplate: The requested template
        
    Raises:
        ValueError: If template name is not found
    """
    if template_name not in ALL_TEMPLATES:
        available = list(ALL_TEMPLATES.keys())
        raise ValueError(f"Template '{template_name}' not found. Available: {available}")
    
    return ALL_TEMPLATES[template_name]


def list_templates() -> List[str]:
    """
    Get list of all available template names.
    
    Returns:
        List[str]: List of template names
    """
    return list(ALL_TEMPLATES.keys())


def get_templates_by_category(category: str) -> Dict[str, PromptTemplate]:
    """
    Get templates by category.
    
    Args:
        category (str): Category name ('qa', 'conversational', 'analysis')
        
    Returns:
        Dict[str, PromptTemplate]: Dictionary of templates in the category
        
    Raises:
        ValueError: If category is not found
    """
    category_map = {
        'qa': QUESTION_ANSWERING_TEMPLATES,
        'conversational': CONVERSATIONAL_TEMPLATES,
        'analysis': ANALYSIS_TEMPLATES
    }
    
    if category not in category_map:
        available = list(category_map.keys())
        raise ValueError(f"Category '{category}' not found. Available: {available}")
    
    return category_map[category]


def create_custom_template(template_str: str, required_vars: Optional[List[str]] = None) -> PromptTemplate:
    """
    Create a custom prompt template.
    
    Args:
        template_str (str): Template string with {variable} placeholders
        required_vars (Optional[List[str]]): List of required variable names
        
    Returns:
        PromptTemplate: Created template
    """
    return PromptTemplate(template_str, required_vars)


def validate_template_variables(template: PromptTemplate, **kwargs) -> tuple:
    """
    Validate variables for a template.
    
    Args:
        template (PromptTemplate): Template to validate
        **kwargs: Variables to check
        
    Returns:
        tuple: (is_valid, missing_variables)
    """
    missing = [var for var in template.required_variables if var not in kwargs]
    return len(missing) == 0, missing


def preview_all_templates() -> Dict[str, str]:
    """
    Generate previews for all templates.
    
    Returns:
        Dict[str, str]: Dictionary mapping template names to their previews
    """
    previews = {}
    for name, template in ALL_TEMPLATES.items():
        try:
            previews[name] = template.preview()
        except Exception as e:
            previews[name] = f"Error generating preview: {e}"
    
    return previews