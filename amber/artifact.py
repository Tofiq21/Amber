"""The ``.amber`` artifact: a single portable, self-certifying file.

Container layout (a zip):
  manifest.json   -- metadata + commitment roots (the public commitment)
  chunks.jsonl    -- one canonical chunk record per line
  vectors.i8.npy  -- (n, dim) int8 quantized embeddings

A leaf binds *both* the chunk source text and its quantized embedding, so the
root commits to the claim: "these embeddings are the image of these chunks
under the pinned embedder."
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .chunk import Chunk, FixedWindowChunker
from .embed import HashEmbedder, quantize
from .merkle import leaf_hash, merkle_root

FORMAT = "amber/1"
SEP = b"\x1f"  # unit separator between chunk bytes and vector bytes


def _leaf_payload(chunk: Chunk, qvec: np.ndarray) -> bytes:
    return chunk.canonical_bytes() + SEP + qvec.tobytes()


def _build_leaves(chunks: List[Chunk], qvecs: np.ndarray) -> List[bytes]:
    n = len(chunks)
    return [leaf_hash(i, n, _leaf_payload(chunks[i], qvecs[i]))
            for i in range(n)]


def _corpus_leaves(chunks: List[Chunk]) -> List[bytes]:
    n = len(chunks)
    return [leaf_hash(i, n, chunks[i].canonical_bytes()) for i in range(n)]


def read_corpus_dir(path: str | Path) -> List[tuple[str, str]]:
    """Read ``*.txt``/``*.md`` files under ``path`` in sorted order."""
    root = Path(path)
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in {".txt", ".md"} and p.is_file())
    out = []
    for p in files:
        out.append((str(p.relative_to(root)),
                    p.read_text(encoding="utf-8", errors="replace")))
    return out


@dataclass
class Artifact:
    manifest: dict
    chunks: List[Chunk]
    qvectors: np.ndarray  # (n, dim) int8

    @property
    def root(self) -> str:
        return self.manifest["merkle_root"]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        buf = io.BytesIO()
        np.save(buf, self.qvectors, allow_pickle=False)
        vec_bytes = buf.getvalue()
        chunk_lines = "\n".join(
            json.dumps(c.to_dict(), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) for c in self.chunks)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json",
                       json.dumps(self.manifest, indent=2, sort_keys=True))
            z.writestr("chunks.jsonl", chunk_lines)
            z.writestr("vectors.i8.npy", vec_bytes)

    @classmethod
    def load(cls, path: str | Path) -> "Artifact":
        with zipfile.ZipFile(path, "r") as z:
            manifest = json.loads(z.read("manifest.json"))
            chunk_text = z.read("chunks.jsonl").decode("utf-8")
            qvectors = np.load(io.BytesIO(z.read("vectors.i8.npy")),
                               allow_pickle=False)
        chunks = []
        for line in chunk_text.splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            chunks.append(Chunk(ordinal=d["ordinal"], source=d["source"],
                                text=d["text"]))
        return cls(manifest=manifest, chunks=chunks, qvectors=qvectors)


def build_artifact(sources: Iterable[tuple[str, str]],
                   embedder=None, chunker=None) -> Artifact:
    """Bank the embedding compute for ``sources`` into an :class:`Artifact`."""
    embedder = embedder or HashEmbedder()
    chunker = chunker or FixedWindowChunker()

    chunks = chunker.chunk_corpus(sources)
    fvecs = embedder.embed([c.text for c in chunks])
    qvecs = quantize(fvecs) if len(fvecs) else np.zeros(
        (0, embedder.dim), dtype=np.int8)

    leaves = _build_leaves(chunks, qvecs)
    corpus_leaves = _corpus_leaves(chunks)

    manifest = {
        "format": FORMAT,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_chunks": len(chunks),
        "dim": int(embedder.dim),
        "embedder": embedder.config(),
        "embedder_config_hash": embedder.config_hash(),
        "chunker": chunker.config(),
        "chunker_config_hash": chunker.config_hash(),
        "merkle_root": merkle_root(leaves).hex(),
        "corpus_root": merkle_root(corpus_leaves).hex(),
    }
    manifest["manifest_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Artifact(manifest=manifest, chunks=chunks, qvectors=qvecs)
