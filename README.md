# AMBER

**Amber by Teresa.** Author: Pete Ferr (@peteferr).
Repo: [github.com/offchainthoughts/amber](https://github.com/offchainthoughts/amber).

Bank the expensive compute once, then carry it offline as a single file that can
prove it is what it claims to be.

AMBER freezes the one-time cost of embedding a corpus into a portable `.amber`
artifact. That file is a cryptographic commitment to the exact corpus and
embedder that produced it. You can hand it to anyone, fully offline, and they
can:

- **integrity-check** it in `O(n)` hashing, with no model, and
- **audit** it in `O(k)` by re-embedding a random sample of `k ≪ n` chunks,
  which proves the stored vectors are the honest image of the source under the
  pinned model without redoing the whole pass.

The artifact plus a small runtime gives offline retrieval where only the short
query is embedded at task time.

## What is new here, and what is not

Offline RAG with a local vector store is prior art. ChromaDB, LanceDB,
sentence-transformers, Ollama, and many tutorials already do it. AMBER does not
claim that as novel. The contribution is the layer on top:

1. A self-certifying artifact format. One file that is a Merkle commitment
   binding source chunks to their quantized embeddings, with domain separation
   and position/length binding.
2. A probabilistic authenticity audit with a proven soundness bound
   (`detection ≥ 1 − (1−ρ)^k`), so banked compute can be trusted at sublinear
   cost.
3. Reproducibility under quantization. Committing to int8 vectors keeps the
   commitment stable despite floating-point nondeterminism across hardware.

The formal treatment, theorems, and threat model are in
[`paper/amber.pdf`](paper/amber.pdf).

## Install

```bash
git clone https://github.com/offchainthoughts/amber
cd amber
pip install -e .              # core: numpy only
pip install -e '.[semantic]'  # optional: sentence-transformers backend
```

## Quickstart

```bash
amber build examples/corpus -o corpus.amber   # bank the compute
amber info   corpus.amber                      # manifest and roots
amber verify corpus.amber                      # O(n) integrity, no model
amber audit  corpus.amber -k 8 --rho 0.1       # O(k) authenticity audit
amber query  corpus.amber "how do plants make sugar"
```

## The point, in one demo

`python examples/demo.py` builds an artifact, then forges half its vectors and
re-commits so the file is internally consistent:

```
4. Forgery: vectors replaced, then RE-COMMITTED
integrity   : True   <- passes, the forgery is self-consistent
audit FAIL: sampled 3, mismatches 2, max |dev| 127 (tol 2)   <- audit catches it
```

Integrity alone is fooled by a self-consistent forgery. The audit is not. That
gap is the reason AMBER exists.

## How it works

| Layer | Object | Cost to verify |
|---|---|---|
| Corpus | `chunks = chunk(D)` | external anchor |
| Bank | `q = quantize(embed(chunk))` | `O(k)` re-embed (audit) |
| Commitment | Merkle root over `(chunk, q)` | `O(n)` full, `O(log n)` proof |

A leaf binds the chunk text and its quantized vector, with the leaf index and
sequence length hashed in. Two roots are published: one over `(chunk, vector)`
pairs binds embedding fidelity, and one over chunks alone binds corpus identity
independently of the embedder.

## Embedders

- `HashEmbedder` (default): signed feature hashing, dependency-free, bitwise
  reproducible (`τ = 0`). Lexical, not semantic. It exists so the
  commitment and audit machinery runs end to end and reproducibly.
- `SentenceTransformerEmbedder` (optional): real semantic quality. Pin the model
  revision. The audit compares within a small int8 tolerance to absorb benign
  float nondeterminism.

## What AMBER does not claim

It binds corpus identity and embedding fidelity. It says nothing about any
downstream generator. Identical retrieved context fed to two models can produce
different answers, so the commitment does not fix task outputs. Banked retrieval
is not banked reasoning. Corpus truthfulness, as opposed to identity, needs an
external trust anchor. See the paper, section 7.

## Tests

```bash
pytest -q
```

## Layout

```
amber/        core library (chunk, embed, merkle, artifact, verify, query, cli)
tests/        unit tests including forgery detection
examples/     small corpus and demo.py
paper/        amber.tex and compiled amber.pdf
```

## License

MIT.
