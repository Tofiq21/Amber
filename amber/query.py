"""Offline retrieval over a banked artifact.

Retrieval reuses the frozen embeddings: only the (short) query is embedded at
task time. Stored int8 vectors are dequantized to float for the cosine score.
This is the cheap "use" side of the bank-once/use-many design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .artifact import Artifact
from .embed import dequantize


@dataclass
class Hit:
    ordinal: int
    source: str
    score: float
    text: str


def search(artifact: Artifact, query: str, embedder, top_k: int = 5) -> List[Hit]:
    if embedder.config_hash() != artifact.manifest["embedder_config_hash"]:
        raise ValueError("embedder config does not match the pinned manifest")
    if len(artifact.chunks) == 0:
        return []
    q = embedder.embed([query])[0]
    mat = dequantize(artifact.qvectors)          # (n, dim) float32, ~unit norm
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    scores = (mat @ q) / norms                   # query is already unit norm
    k = min(top_k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    hits = []
    for i in top:
        c = artifact.chunks[int(i)]
        hits.append(Hit(ordinal=c.ordinal, source=c.source,
                        score=float(scores[i]), text=c.text))
    return hits
