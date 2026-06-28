# Audit

Full review of the AMBER reference implementation and paper. Date: 2026-06-28.

## Method

Three layers were checked: the unit suite, an edge-case and security harness,
and a paper-to-code consistency pass that re-derives the committed leaf format
from the paper's Definition and compares it byte-for-byte against the code.

## Results

All 8 unit tests pass. All 21 audit checks pass.

### Unit suite (`tests/test_amber.py`)
- determinism of the build (same corpus and params give the same root)
- save/load round-trip preserves root and vectors
- integrity detects a one-bit vector tamper
- integrity detects a text tamper
- clean audit passes with max deviation 0 (deterministic embedder)
- audit catches forged-then-recommitted vectors that integrity cannot
- Merkle inclusion proofs verify and reject wrong leaves
- retrieval returns the on-topic source

### Edge cases
- empty corpus: 0 chunks, integrity holds, audit vacuously passes, search empty
- single chunk: root equals the lone leaf, empty inclusion proof verifies
- odd leaf count (3): proofs verify at every index; promote-odd is consistent
- `quantize` maps the extremes to +/-127 and stays within int8 (no overflow)
- `top_k` and `k` larger than `n` are clamped, not errors
- audit and search reject an embedder whose config does not match the manifest

### Paper-to-code consistency
- leaf hash equals `H(0x00 || i || n || |p| || p)` with
  `p = canon(c) || 0x1f || q`, re-derived independently and matched
- leaf tag `0x00`, node tag `0x01`, separator `0x1f` as specified
- two distinct roots present: `merkle_root` over (chunk, vector) and
  `corpus_root` over chunks alone
- audit detection bound equals `1 - (1 - rho)^k`; reported miss probability
  equals `(1 - rho)^k`

### Security sanity
- domain separation holds: an internal node hash cannot be reinterpreted as a
  leaf, because the leaf preimage carries a distinct tag plus index and length
- the deterministic default embedder is bitwise reproducible (tau = 0), so the
  quantized commitment is exactly regenerable

### Metadata
- zero em-dashes repo-wide; zero placeholder handles
- author Pete Ferr (@peteferr) consistent across paper, `pyproject.toml`, and
  `LICENSE`
- paper typeset in Times (`mathptmx`) to match the reference style

## Known limitations (by design, see paper section 9)
- the default embedder is lexical, not semantic; use the transformer backend for
  retrieval quality, which moves tau from 0 to a small tolerance
- the audit certifies fidelity to a pinned map, not corpus truthfulness;
  corpus authenticity requires an external anchor for `corpus_root`
- `manifest_id` is an informational digest and is not re-checked by `verify`
