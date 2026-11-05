import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

messages = [
    "How do I reset my password?",
    "I forgot my login information.",
    "What are your business hours?",
]

embeddings = []

for message in messages:
    embedding = client.feature_extraction(
        message,
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

    embeddings.append({
        "text": message,
        "embedding": embedding,
    })

print(f"Created {len(embeddings)} embeddings.")
