# embedding_cache.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from cost_tracker import CostTracker
from openai import OpenAI

EMBEDDING_MODEL: str = "text-embedding-3-small"


# -----------------------------
# 1) A tiny embedding model API
# -----------------------------
class SimpleEmbeddingModel:
    """
    Very small wrapper around an embedding backend.
    provider='gpt' uses OpenAI embeddings (requires OPENAI_API_KEY).
    provider='mock' uses a deterministic hashing trick (offline).
    """
    def __init__(self, provider: str = "gpt", model: str = EMBEDDING_MODEL, dim: int = 1536,
                 tracker: CostTracker | None = None):
        self.provider = provider.lower()
        self.model = model
        self.dim = dim  # used for mock mode
        self.tracker = tracker  # <- NEW

        if self.provider == "gpt":
            try:
                self._client = OpenAI()
            except Exception as e:
                raise RuntimeError(
                    "Install and configure the OpenAI client (pip install openai). "
                    "Also set OPENAI_API_KEY in the environment."
                ) from e
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @property
    def id(self) -> str:
        if self.provider == "gpt":
            return f"gpt::{self.model}"
        return f"mock::{self.dim}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "gpt":
            return self._embed_gpt(texts)
        raise NotImplementedError("Mock provider not implemented")

    # --- Backends ---
    def _embed_gpt(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)

        # Robustly extract token usage (API names vary over time)
        usage = getattr(resp, "usage", None)
        input_tokens = 0
        if usage:
            # Try a few common fields
            input_tokens = (
                getattr(usage, "prompt_tokens", None)
                or getattr(usage, "input_tokens", None)
                or getattr(usage, "total_tokens", 0)
            ) or 0

        # Log to tracker (embeddings charge on input only)
        if self.tracker is not None:
            meta = {"n_inputs": len(texts)}
            self.tracker.log(kind="embedding", model=self.model, input_tokens=int(input_tokens), output_tokens=0, meta=meta)

        return [d.embedding for d in resp.data]


# ---------------------------------------
# 2) Persistent cache with local JSON file
# ---------------------------------------
@dataclass
class EmbeddingCache:
    """
    Persistent file-backed cache for text embeddings.
    - Key = SHA256(canonical_text) scoped by model id
    - Value = embedding vector (list[float])
    File format (JSON):
    {
      "version": 1,
      "model_id": "<model id>",
      "items": {"<key>": {"text": "<original>", "vector": [...]}, ...}
    }
    Notes:
      * Cache is scoped per model_id. A new model_id creates a fresh namespace.
      * Canonicalization collapses whitespace to keep keys stable.
    """
    cache_path: str = "embeddings_cache.json"
    model: Optional[SimpleEmbeddingModel] = None
    autosave: bool = True
    new_embed_count: int = 0

    def __post_init__(self):
        if self.model is None:
            # Default to GPT embeddings
            self.model = SimpleEmbeddingModel(provider="gpt", model="text-embedding-3-small")
        self._data: Dict[str, Dict] = {"version": 1, "model_id": self.model.id, "items": {}}
        self._load_or_init()

    # --- Public API ---
    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text (cached)."""
        vectors = self.embed_many([text])
        return vectors[0]

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with caching; preserves input order."""
        new_emb_count = 0
        canon_texts = [self._canonicalize(t) for t in texts]
        keys = [self._key(ct) for ct in canon_texts]

        # Find cache hits & misses
        hits: Dict[int, List[float]] = {}
        misses: List[Tuple[int, str]] = []  # (index, canonical_text)
        for i, (k, ct) in enumerate(zip(keys, canon_texts)):
            item = self._data["items"].get(k)
            if item is not None:
                hits[i] = item["vector"]
            else:
                misses.append((i, ct))

        # Compute for misses (if any)
        if misses:
            # Call model with original canonical text (not the key)
            to_compute = [ct for _, ct in misses]
            new_vecs = self.model.embed(to_compute)
            new_emb_count += len(new_vecs)
            for (idx, ct), vec in zip(misses, new_vecs):
                k = self._key(ct)
                # Save original text (optional; here we store canonical, but could add original too)
                self._data["items"][k] = {"text": ct, "vector": vec}
                hits[idx] = vec
            if self.autosave:
                self.save()

        # Re-order to match input
        self.new_embed_count += new_emb_count
        return [hits[i] for i in range(len(texts))]

    def save(self):
        """Persist cache to disk."""
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)
        os.replace(tmp_path, self.cache_path)

    def clear(self):
        """Drop all cached items for the current model id."""
        self._data = {"version": 1, "model_id": self.model.id, "items": {}}
        if self.autosave:
            self.save()

    # --- Private helpers ---
    def _load_or_init(self):
        if not os.path.exists(self.cache_path):
            # fresh file
            if self.autosave:
                self.save()
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Corrupt file; start clean
            if self.autosave:
                self.save()
            return

        # If file belongs to another model id, keep both by namespacing inside 'items'
        # To keep it simple, we reset unless it matches the current model id.
        if isinstance(data, dict) and data.get("model_id") == self.model.id:
            self._data = data
        else:
            # Different model id or unknown format -> re-init
            self._data = {"version": 1, "model_id": self.model.id, "items": {}}
            if self.autosave:
                self.save()

    @staticmethod
    def _canonicalize(text: str) -> str:
        # collapse whitespace + strip
        return " ".join(text.split()).strip().lower()

    def _key(self, canonical_text: str) -> str:
        h = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
        return f"{self.model.id}:{h}"


# -----------------------------
# 3) Tiny usage example
# -----------------------------

def get_embedding_model(tracker: CostTracker, cache_path: str = "embeddings_cache.json") -> EmbeddingCache:
    model = SimpleEmbeddingModel(provider="gpt",  model=EMBEDDING_MODEL, tracker=tracker)
    cache = EmbeddingCache(cache_path=cache_path, model=model, autosave=True)
    return cache

def _test():
    # Choose one:
    model = SimpleEmbeddingModel(provider="gpt",  model="text-embedding-3-small")
    # model = SimpleEmbeddingModel(provider="mock", dim=128)  # offline testing

    cache = EmbeddingCache(cache_path="embeddings_cache.json", model=model, autosave=True)

    texts = [
        "KADTVSLGH — looks functional due to KR start and H near C-terminus.",
        "RVEGSATHI — acidic at pos3, contains G, and ends in I.",
        "RVEGSATHI — acidic at pos3, contains G, and ends in I.",  # duplicate to test cache hit
    ]

    vecs = cache.embed_many(texts)
    print("Got vectors:", [len(v) for v in vecs])
    print("New embeddings:", cache.new_embed_count)

    # Single lookup (should hit cache now)
    v = cache.embed("RVEGSATHI — acidic at pos3, contains G, and ends in I.")
    print("Single vector dim:", len(v))


if __name__ == "__main__":
    _test()