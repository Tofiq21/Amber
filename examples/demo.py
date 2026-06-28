"""End-to-end demo: build, verify, audit (honest vs forged), and query.

Run:  python examples/demo.py
"""

import numpy as np

from amber import build_artifact, HashEmbedder, verify_integrity, audit, search
from amber.artifact import read_corpus_dir, _build_leaves, _corpus_leaves
from amber.merkle import merkle_root


def banner(s):
    print("\n" + "=" * 60 + f"\n{s}\n" + "=" * 60)


def main():
    emb = HashEmbedder(dim=256, seed=0)
    sources = read_corpus_dir("examples/corpus")
    art = build_artifact(sources, embedder=emb)

    banner("1. Banked artifact")
    print("chunks      :", art.manifest["n_chunks"])
    print("merkle_root :", art.manifest["merkle_root"][:32], "...")

    banner("2. Integrity + honest audit")
    print("integrity   :", verify_integrity(art))
    print(audit(art, emb, k=art.manifest["n_chunks"], tolerance=0).summary())

    banner("3. Offline retrieval (only the query is embedded)")
    for h in search(art, "verify large data with a single hash", emb, top_k=2):
        print(f"  [{h.score:+.3f}] {h.source}#{h.ordinal}")

    banner("4. Forgery: vectors replaced, then RE-COMMITTED")
    rng = np.random.default_rng(1)
    n = art.manifest["n_chunks"]
    for i in range(0, n, 2):                      # corrupt half
        art.qvectors[i] = rng.integers(-127, 128, art.qvectors.shape[1]
                                       ).astype(np.int8)
    art.manifest["merkle_root"] = merkle_root(
        _build_leaves(art.chunks, art.qvectors)).hex()
    art.manifest["corpus_root"] = merkle_root(_corpus_leaves(art.chunks)).hex()

    print("integrity   :", verify_integrity(art), "  <- passes, the forgery is self-consistent")
    rep = audit(art, emb, k=min(8, n), tolerance=2, hypothesize_rho=0.5)
    print(rep.summary(), "  <- audit re-embeds and catches it")


if __name__ == "__main__":
    main()
