"""
Sentence-transformer wrapper for encoding texts into vectors.

Uses a multilingual model so Italian plans can match English grants
without any translation step.
"""

import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_MODEL_NAME,
    GRANTS_DB_PATH,
    PLANS_DB_PATH,
    PROCESSED_DATA_DIR,
)
from src.data_collection.loader import load_grants, load_plans
from src.preprocessing.text_cleaner import clean_text, chunk_text


class Embedder:

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_long_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> np.ndarray:
        """
        For documents that exceed the model's context window:
        chunk → embed each → average → renormalize.
        """
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned, chunk_size, overlap)
        chunk_vectors = self.embed_texts(chunks)

        avg = chunk_vectors.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm
        return avg


def build_grant_embeddings(embedder: Embedder | None = None) -> dict:
    """Embed all grants and dump to disk. Returns the result dict."""
    embedder = embedder or Embedder()
    grants = load_grants()

    print(f"Embedding {len(grants)} grants...")
    vectors = embedder.embed_texts([clean_text(g.description) for g in grants])

    result = {
        "ids": [g.id for g in grants],
        "vectors": vectors,
        "model": embedder.model_name,
    }

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRANTS_DB_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"  → saved to {GRANTS_DB_PATH}")

    return result


def build_plan_embeddings(embedder: Embedder | None = None) -> dict:
    """Embed all plans (long docs, so we chunk+average) and dump to disk."""
    embedder = embedder or Embedder()
    plans = load_plans()

    print(f"Embedding {len(plans)} plans...")
    vectors = []
    for p in plans:
        vectors.append(embedder.embed_long_text(p.full_text))
        print(f"  ✓ {p.municipality}")

    result = {
        "ids": [p.id for p in plans],
        "vectors": np.stack(vectors),
        "model": embedder.model_name,
    }

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLANS_DB_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"  → saved to {PLANS_DB_PATH}")

    return result


def load_grant_embeddings() -> dict:
    with open(GRANTS_DB_PATH, "rb") as f:
        return pickle.load(f)


def load_plan_embeddings() -> dict:
    with open(PLANS_DB_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    emb = Embedder()

    # sanity: cross-lingual similarity should be high,
    # unrelated topics should be low
    v1 = emb.embed_text("Installazione di pannelli solari sugli edifici comunali")
    v2 = emb.embed_text("Solar panel deployment on public buildings")
    v3 = emb.embed_text("Medieval castle restoration project")

    print(f"IT↔EN solar panels: {np.dot(v1, v2):.3f}")
    print(f"solar vs castles:   {np.dot(v1, v3):.3f}")
    print()

    build_grant_embeddings(emb)
    build_plan_embeddings(emb)
