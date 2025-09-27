from rag_lib.models.embedding import SBERTEmbedding

model = SBERTEmbedding(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True,
    batch_size=16,
    max_seq_length=384,
    similarity_type="cosine"
)

text = "Hello, world!"  

print(model.get_token_count(text))
print(model.encode_text(text))