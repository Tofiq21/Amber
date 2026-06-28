"""Verification: cheap integrity check, and the probabilistic audit.

Two guarantees, at very different costs:

* ``verify_integrity`` recomputes the Merkle root from the stored chunks and
  vectors. It needs no model and runs in O(n) hashing. It detects any
  post-hoc tampering of a committed artifact, but it does *not* prove the
  banked embeddings were honestly computed -- a forger could commit to
  garbage vectors consistently.

* ``audit`` closes that gap. It re-embeds a random sample of k chunks with the
  pinned embedder and checks each against the committed vector within a
  tolerance. If a fraction rho of leaves are inconsistent with the pinned map,
  the audit detects it with probability at least 1 - (1 - rho)^k. This is the
  core contribution: trust in the banked compute at cost O(k), k << n.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .artifact import Artifact, _build_leaves, _corpus_leaves
from .embed import quantize
from .merkle import merkle_root


def verify_integrity(artifact: Artifact) -> bool:
    """Recompute roots from stored data; compare to the manifest."""
    leaves = _build_leaves(artifact.chunks, artifact.qvectors)
    if merkle_root(leaves).hex() != artifact.manifest["merkle_root"]:
        return False
    corpus = _corpus_leaves(artifact.chunks)
    return merkle_root(corpus).hex() == artifact.manifest["corpus_root"]


@dataclass
class AuditReport:
    sampled: int
    mismatches: List[int] = field(default_factory=list)
    max_abs_dev: int = 0
    tolerance: int = 0
    detection_bound: Optional[float] = None  # for a hypothesized rho

    @property
    def passed(self) -> bool:
        return len(self.mismatches) == 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        s = (f"audit {status}: sampled {self.sampled}, "
             f"mismatches {len(self.mismatches)}, "
             f"max |dev| {self.max_abs_dev} (tol {self.tolerance})")
        if self.detection_bound is not None:
            s += f"\n  if rho>=hypothesis, miss prob <= {1 - self.detection_bound:.3e}"
        return s


def audit(artifact: Artifact, embedder, k: int = 32, tolerance: int = 2,
          rng: Optional[np.random.Generator] = None,
          hypothesize_rho: Optional[float] = None) -> AuditReport:
    """Re-embed k random chunks and compare to the committed vectors.

    ``tolerance`` is the maximum allowed per-component absolute deviation in
    int8 units (absorbs benign floating-point nondeterminism for semantic
    backends; for the deterministic HashEmbedder, deviation is 0).
    """
    n = len(artifact.chunks)
    if n == 0:
        return AuditReport(sampled=0, tolerance=tolerance)
    if embedder.config_hash() != artifact.manifest["embedder_config_hash"]:
        raise ValueError("embedder config does not match the pinned manifest")

    k = min(k, n)
    # Cryptographically strong, unbiased sample without replacement.
    if rng is None:
        idx = _secure_sample(n, k)
    else:
        idx = list(rng.choice(n, size=k, replace=False))

    texts = [artifact.chunks[i].text for i in idx]
    recomputed = quantize(embedder.embed(texts))

    mismatches: List[int] = []
    max_dev = 0
    for j, i in enumerate(idx):
        dev = int(np.max(np.abs(recomputed[j].astype(np.int16)
                                - artifact.qvectors[i].astype(np.int16))))
        max_dev = max(max_dev, dev)
        if dev > tolerance:
            mismatches.append(int(i))

    bound = None
    if hypothesize_rho is not None:
        bound = 1.0 - (1.0 - hypothesize_rho) ** k

    return AuditReport(sampled=k, mismatches=mismatches, max_abs_dev=max_dev,
                       tolerance=tolerance, detection_bound=bound)


def _secure_sample(n: int, k: int) -> List[int]:
    """Uniform sample of k distinct indices in [0, n) using os entropy."""
    chosen = set()
    while len(chosen) < k:
        chosen.add(secrets.randbelow(n))
    return sorted(chosen)
