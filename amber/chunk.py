"""Deterministic chunking.

A chunker is a pure function of (corpus bytes, parameters). Determinism is a
hard requirement: the artifact commits to the chunk boundaries, so two builders
given the same corpus and parameters must produce byte-identical chunks.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Iterable, List


_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    """One unit of banked context.

    Attributes
    ----------
    ordinal : global position of the chunk in the corpus (0-based).
    source  : logical source identifier (e.g. relative file path).
    text    : the normalized chunk text.
    """

    ordinal: int
    source: str
    text: str

    def canonical_bytes(self) -> bytes:
        """Canonical, stable serialization used inside the commitment."""
        obj = {"o": self.ordinal, "s": self.source, "t": self.text}
        return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    """NFC-normalize, strip, and collapse internal whitespace.

    Normalization is part of the committed definition; changing it changes the
    corpus identity. We keep it intentionally minimal and explicit.
    """
    text = unicodedata.normalize("NFC", text)
    text = _WS.sub(" ", text)
    return text.strip()


class FixedWindowChunker:
    """Split each source into fixed word-count windows with overlap.

    Parameters
    ----------
    window : number of words per chunk.
    overlap : number of overlapping words between consecutive chunks.
    """

    def __init__(self, window: int = 120, overlap: int = 20):
        if window <= 0:
            raise ValueError("window must be positive")
        if not (0 <= overlap < window):
            raise ValueError("overlap must satisfy 0 <= overlap < window")
        self.window = window
        self.overlap = overlap

    # -- params participate in the commitment via this stable descriptor ----
    def config(self) -> dict:
        return {"chunker": "fixed-window", "window": self.window,
                "overlap": self.overlap, "normalize": "nfc+wscollapse"}

    def config_hash(self) -> str:
        blob = json.dumps(self.config(), sort_keys=True,
                          separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def chunk_source(self, source: str, text: str) -> List[str]:
        words = normalize(text).split(" ")
        if words == [""]:
            return []
        step = self.window - self.overlap
        out: List[str] = []
        i = 0
        n = len(words)
        while i < n:
            out.append(" ".join(words[i:i + self.window]))
            if i + self.window >= n:
                break
            i += step
        return out

    def chunk_corpus(self, sources: Iterable[tuple[str, str]]) -> List[Chunk]:
        """``sources`` is an iterable of (source_id, text), consumed in the
        order given. The caller is responsible for a deterministic order
        (e.g. sorted file paths)."""
        chunks: List[Chunk] = []
        ordinal = 0
        for source, text in sources:
            for piece in self.chunk_source(source, text):
                chunks.append(Chunk(ordinal=ordinal, source=source, text=piece))
                ordinal += 1
        return chunks
