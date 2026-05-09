import os
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env") # Assuming .env is in parent directory or same root

class APISearchService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API Key not found. Please set OPENAI_API_KEY in .env")
        self.client = OpenAI(api_key=self.api_key)

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate an embedding for a piece of text using OpenAI API."""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def get_embeddings_batch(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
