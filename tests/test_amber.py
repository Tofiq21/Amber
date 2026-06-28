"""Tests: determinism, integrity, tamper-detection, audit soundness, retrieval."""

import numpy as np

from amber import (FixedWindowChunker, HashEmbedder, build_artifact,
                   verify_integrity, audit, search, Artifact)
from amber.merkle import merkle_root, merkle_proof, verify_proof, leaf_hash


CORPUS = [
    ("a.txt", "The quick brown fox jumps over the lazy dog. " * 30),
    ("b.txt", "Banked compute is frozen into a portable artifact. " * 30),
    ("c.txt", "Retrieval reuses embeddings computed once, offline. " * 30),
]


def _build():
    return build_artifact(CORPUS, embedder=HashEmbedder(dim=128, seed=7),
                          chunker=FixedWindowChunker(window=40, overlap=8))


def test_determinism():
    a1, a2 = _build(), _build()
    assert a1.manifest["merkle_root"] == a2.manifest["merkle_root"]
    assert a1.manifest["corpus_root"] == a2.manifest["corpus_root"]
    assert np.array_equal(a1.qvectors, a2.qvectors)


def test_roundtrip(tmp_path):
    a = _build()
    p = tmp_path / "x.amber"
    a.save(p)
    b = Artifact.load(p)
    assert b.manifest["merkle_root"] == a.manifest["merkle_root"]
    assert np.array_equal(a.qvectors, b.qvectors)
    assert verify_integrity(b)


def test_integrity_detects_vector_tamper():
    a = _build()
    a.qvectors[3, 0] = np.int8(int(a.qvectors[3, 0]) ^ 0x1)  # flip one unit
    assert verify_integrity(a) is False


def test_integrity_detects_text_tamper():
    a = _build()
    c = a.chunks[2]
    a.chunks[2] = type(c)(ordinal=c.ordinal, source=c.source,
                          text=c.text + " TAMPER")
    assert verify_integrity(a) is False


def test_audit_passes_clean():
    a = _build()
    emb = HashEmbedder(dim=128, seed=7)
    rep = audit(a, emb, k=len(a.chunks), tolerance=0)
    assert rep.passed
    assert rep.max_abs_dev == 0  # deterministic embedder => exact


def test_audit_catches_forged_vectors():
    a = _build()
    # Forge: replace half the committed vectors with noise, then re-commit so
    # integrity still passes. Only an audit (re-embedding) can catch this.
    rng = np.random.default_rng(0)
    n = len(a.chunks)
    forged = list(range(0, n, 2))
    for i in forged:
        a.qvectors[i] = (rng.integers(-127, 128, size=a.qvectors.shape[1])
                         ).astype(np.int8)
    # Re-commit so verify_integrity cannot see the forgery.
    from amber.artifact import _build_leaves, _corpus_leaves
    a.manifest["merkle_root"] = merkle_root(
        _build_leaves(a.chunks, a.qvectors)).hex()
    a.manifest["corpus_root"] = merkle_root(_corpus_leaves(a.chunks)).hex()
    assert verify_integrity(a)  # consistent, but dishonest

    emb = HashEmbedder(dim=128, seed=7)
    caught = 0
    trials = 40
    for _ in range(trials):
        rep = audit(a, emb, k=8, tolerance=2)
        if not rep.passed:
            caught += 1
    # rho = 0.5, k = 8 => miss prob (1-0.5)^8 ~ 0.0039 per trial; expect ~all.
    assert caught >= trials - 1


def test_merkle_proof():
    leaves = [leaf_hash(i, 5, f"x{i}".encode()) for i in range(5)]
    root = merkle_root(leaves)
    for i in range(5):
        pf = merkle_proof(leaves, i)
        assert verify_proof(leaves[i], i, pf, root)
        assert not verify_proof(leaves[(i + 1) % 5], i, pf, root)


def test_search_finds_topic():
    a = _build()
    emb = HashEmbedder(dim=128, seed=7)
    hits = search(a, "portable banked compute frozen artifact", emb, top_k=1)
    assert hits and hits[0].source == "b.txt"
