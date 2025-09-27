from .__version__ import __version__
from .models.embedding import EmbeddingModel, OpenAIEmbedding
from .schemas.schema import Query, Document

# Lazy import for optional dependencies
def __getattr__(name):
    if name == 'SBERTEmbedding':
        from .models.embedding import SBERTEmbedding
        return SBERTEmbedding
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['__version__', 'EmbeddingModel', 'OpenAIEmbedding', 'SBERTEmbedding', 'Query', 'Document']