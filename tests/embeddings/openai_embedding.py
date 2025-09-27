from rag_lib.models.embedding import OpenAIEmbedding
from dotenv import load_dotenv
import os

load_dotenv(".env", override=True)

# Example usage
embedding_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    rate_limit_delay=0.1,
    max_retries=3,
    timeout=30.0,
)

text = "Hello, world!"
embedding = embedding_model.encode_text(text)

print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")