"""Embedders and fixed-point quantization.

The embedder is *pinned*: its identity and configuration are committed inside
the artifact manifest, so an auditor re-embeds with the same map. We ship a
fully deterministic feature-hashing embedder (no model download, bitwise
reproducible) as the default. A semantic embedder (sentence-transformers) is
available as an optional drop-in; see ``SentenceTransformerEmbedder``.

Quantization to fixed-point int8 is what makes the *commitment* exactly
reproducible: floating-point embedding is not bitwise stable across hardware,
but the quantized vector is stable up to a controlled tolerance, and that
quantized vector is what we commit to.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import List

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")
_QSCALE = 127.0


def quantize(vecs: np.ndarray) -> np.ndarray:
    """Map L2-normalized float vectors (components in [-1, 1]) to int8."""
    q = np.rint(np.clip(vecs, -1.0, 1.0) * _QSCALE)
    return q.astype(np.int8)


def dequantize(q: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize` (lossy)."""
    return q.astype(np.float32) / _QSCALE


class HashEmbedder:
    """Signed feature-hashing embedder. Deterministic and dependency-free.

    Tokens are lowercased ``[a-z0-9]+`` runs. Each token is hashed (BLAKE2b,
    keyed by ``seed``) to a bucket index and a sign; term frequencies accumulate
    into the bucket; the vector is L2-normalized. This is a bag-of-words
    baseline: it captures lexical overlap, not deep semantics. It exists so the
    artifact/commitment/audit machinery runs end-to-end and reproducibly; swap
    in a semantic embedder for retrieval quality.
    """

    name = "hash-fh-v1"

    def __init__(self, dim: int = 256, seed: int = 0):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.seed = seed
        self._key = seed.to_bytes(8, "little", signed=False)

    def config(self) -> dict:
        return {"embedder": self.name, "dim": self.dim, "seed": self.seed,
                "quant": "int8", "qscale": _QSCALE}

    def config_hash(self) -> str:
        blob = json.dumps(self.config(), sort_keys=True,
                          separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def _bucket_sign(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(token.encode("utf-8"), key=self._key,
                            digest_size=8).digest()
        v = int.from_bytes(h, "little")
        bucket = v % self.dim
        sign = 1.0 if (v >> 63) & 1 else -1.0
        return bucket, sign

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in _TOK.findall(text.lower()):
            b, s = self._bucket_sign(tok)
            vec[b] += s
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_one(t) for t in texts], axis=0)


class SentenceTransformerEmbedder:
    """Optional semantic backend. Requires ``sentence-transformers`` and a
    one-time model download (not available in restricted/offline build envs).

    Note on reproducibility: transformer inference is not guaranteed to be
    bitwise-identical across hardware. Quantization to int8 absorbs small
    perturbations, and the auditor compares within a tolerance (see verify.py).
    Pin the model revision for a stable commitment.
    """

    name = "sentence-transformers"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 revision: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "Install extras: pip install 'amber-kb[semantic]'") from e
        self.model_id = model
        self.revision = revision
        self._m = SentenceTransformer(model, revision=revision, device="cpu")
        self.dim = self._m.get_sentence_embedding_dimension()

    def config(self) -> dict:
        return {"embedder": self.name, "model": self.model_id,
                "revision": self.revision, "dim": self.dim,
                "quant": "int8", "qscale": _QSCALE}

    def config_hash(self) -> str:
        blob = json.dumps(self.config(), sort_keys=True,
                          separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def embed(self, texts: List[str]) -> np.ndarray:  # pragma: no cover
        import numpy as _np
        v = self._m.encode(texts, normalize_embeddings=True,
                           convert_to_numpy=True)
        return v.astype(_np.float32)
