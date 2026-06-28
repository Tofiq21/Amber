"""Binary Merkle tree over artifact leaves, with domain separation.

Leaf and internal-node hashes use distinct tag bytes so a leaf can never be
reinterpreted as an internal node (a classic Merkle second-preimage defense).
Each leaf preimage also binds the leaf index and the total leaf count, so the
position and length of the committed sequence are fixed by the root.
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

LEAF_TAG = b"\x00"
NODE_TAG = b"\x01"


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


def leaf_hash(index: int, total: int, payload: bytes) -> bytes:
    """Hash a leaf, binding its index and the sequence length."""
    return _h(LEAF_TAG,
              index.to_bytes(8, "big"),
              total.to_bytes(8, "big"),
              len(payload).to_bytes(8, "big"),
              payload)


def _node_hash(left: bytes, right: bytes) -> bytes:
    return _h(NODE_TAG, left, right)


def _levels(leaves: List[bytes]) -> List[List[bytes]]:
    if not leaves:
        return [[_h(NODE_TAG, b"empty")]]
    levels = [leaves]
    cur = leaves
    while len(cur) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(cur), 2):
            left = cur[i]
            right = cur[i + 1] if i + 1 < len(cur) else cur[i]  # promote odd
            nxt.append(_node_hash(left, right))
        levels.append(nxt)
        cur = nxt
    return levels


def merkle_root(leaves: List[bytes]) -> bytes:
    """Root over a list of leaf hashes."""
    return _levels(leaves)[-1][0]


def merkle_proof(leaves: List[bytes], index: int) -> List[Tuple[str, bytes]]:
    """Inclusion proof for ``leaves[index]`` as a list of (side, sibling)."""
    if not (0 <= index < len(leaves)):
        raise IndexError("leaf index out of range")
    levels = _levels(leaves)
    proof: List[Tuple[str, bytes]] = []
    idx = index
    for level in levels[:-1]:
        if idx % 2 == 0:
            sib = level[idx + 1] if idx + 1 < len(level) else level[idx]
            proof.append(("R", sib))
        else:
            proof.append(("L", level[idx - 1]))
        idx //= 2
    return proof


def verify_proof(leaf: bytes, index: int, proof: List[Tuple[str, bytes]],
                 root: bytes) -> bool:
    """Recompute the root from a leaf and its inclusion proof."""
    acc = leaf
    for side, sib in proof:
        acc = _node_hash(sib, acc) if side == "L" else _node_hash(acc, sib)
    return acc == root
