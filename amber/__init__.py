"""AMBER: Auditable, Memory-Banked Embedding Retrieval.

A portable, content-addressed, self-verifying offline knowledge artifact.

The expensive computation (embedding a corpus) is performed once and frozen
into a single ``.amber`` file that is a cryptographic commitment to the exact
corpus and embedder that produced it. The artifact can be:

  * integrity-checked in O(n) hashing with no model, and
  * probabilistically *audited* against the source by re-embedding a small
    random sample, proving the banked compute is authentic without redoing
    the whole pass.

See the accompanying paper (paper/amber.tex) for the formal treatment.
"""

from .chunk import Chunk, FixedWindowChunker
from .embed import HashEmbedder, quantize, dequantize
from .merkle import merkle_root, merkle_proof, verify_proof
from .artifact import build_artifact, Artifact
from .verify import verify_integrity, audit, AuditReport
from .query import search

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "FixedWindowChunker",
    "HashEmbedder",
    "quantize",
    "dequantize",
    "merkle_root",
    "merkle_proof",
    "verify_proof",
    "build_artifact",
    "Artifact",
    "verify_integrity",
    "audit",
    "AuditReport",
    "search",
    "__version__",
]
