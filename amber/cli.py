"""AMBER command line: build / info / verify / audit / query."""

from __future__ import annotations

import argparse
import json
import sys

from .artifact import Artifact, build_artifact, read_corpus_dir
from .embed import HashEmbedder
from .verify import verify_integrity, audit
from .query import search


def _embedder(args):
    return HashEmbedder(dim=args.dim, seed=args.seed)


def cmd_build(args):
    sources = read_corpus_dir(args.corpus)
    art = build_artifact(sources, embedder=_embedder(args))
    art.save(args.out)
    print(f"built {args.out}")
    print(f"  chunks      : {art.manifest['n_chunks']}")
    print(f"  merkle_root : {art.manifest['merkle_root']}")
    print(f"  corpus_root : {art.manifest['corpus_root']}")


def cmd_info(args):
    art = Artifact.load(args.artifact)
    print(json.dumps(art.manifest, indent=2, sort_keys=True))


def cmd_verify(args):
    art = Artifact.load(args.artifact)
    ok = verify_integrity(art)
    print("integrity:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_audit(args):
    art = Artifact.load(args.artifact)
    emb = HashEmbedder(dim=art.manifest["embedder"]["dim"],
                       seed=art.manifest["embedder"]["seed"])
    rep = audit(art, emb, k=args.k, tolerance=args.tol,
                hypothesize_rho=args.rho)
    print(rep.summary())
    sys.exit(0 if rep.passed else 1)


def cmd_query(args):
    art = Artifact.load(args.artifact)
    emb = HashEmbedder(dim=art.manifest["embedder"]["dim"],
                       seed=art.manifest["embedder"]["seed"])
    hits = search(art, args.text, emb, top_k=args.k)
    for h in hits:
        print(f"[{h.score:+.3f}] {h.source}#{h.ordinal}: {h.text[:160]}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="amber", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="bank a corpus into a .amber artifact")
    b.add_argument("corpus")
    b.add_argument("-o", "--out", default="corpus.amber")
    b.add_argument("--dim", type=int, default=256)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("info", help="print the manifest")
    i.add_argument("artifact")
    i.set_defaults(func=cmd_info)

    v = sub.add_parser("verify", help="integrity check (no model)")
    v.add_argument("artifact")
    v.set_defaults(func=cmd_verify)

    a = sub.add_parser("audit", help="probabilistic authenticity audit")
    a.add_argument("artifact")
    a.add_argument("-k", type=int, default=32, help="sample size")
    a.add_argument("--tol", type=int, default=2, help="int8 tolerance")
    a.add_argument("--rho", type=float, default=None,
                   help="hypothesized tamper fraction for the detection bound")
    a.set_defaults(func=cmd_audit)

    q = sub.add_parser("query", help="offline retrieval")
    q.add_argument("artifact")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=5)
    q.set_defaults(func=cmd_query)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except BrokenPipeError:  # e.g. piping into `head`
        try:
            sys.stdout.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
