"""OpenAI-compatible embedding client for matching and retrieval."""

import os
import time
import numpy as np
from typing import List, Optional, Union
from openai import OpenAI


class EmbeddingClient:
    """Create vectors for exemplar retrieval and semantic matching."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):

        if api_key is None:
            api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(
                f"API key not provided. Set {api_key_env} environment variable."
            )

        self.client = OpenAI(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key
        )
        self.model = model
        self.dimension = dimension
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def embed_text(self, text: str) -> np.ndarray:

        if not text or not text.strip():

            return np.zeros(self.dimension, dtype=np.float32)

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text.strip()
                )

                embedding = response.data[0].embedding
                return np.array(embedding, dtype=np.float32)

            except Exception as e:
                print(f"[EmbeddingClient] Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

        return np.zeros(self.dimension, dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        embeddings = []

        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        return np.stack(embeddings)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> np.ndarray:

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.append(batch_embeddings)

            if i + batch_size < len(texts):
                time.sleep(0.5)

        return np.vstack(all_embeddings)

if __name__ == "__main__":

    try:
        client = EmbeddingClient()

        test_text = "Patient presents with fever and productive cough"
        embedding = client.embed_text(test_text)

        print(f"Text: {test_text}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"First 5 values: {embedding[:5]}")

    except Exception as e:
        print(f"Test failed: {e}")

