import os
from huggingface_hub import InferenceClient

# Create the Hugging Face client
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

# Text you want to convert into an embedding
text = "Hello! I need help with my account."

# Create an embedding
embedding = client.feature_extraction(
    text,
    model="sentence-transformers/all-MiniLM-L6-v2",
)

print("Embedding:")
print(embedding)

print("\nEmbedding dimensions:")
print(len(embedding))
