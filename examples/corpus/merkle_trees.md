# Merkle Trees
A Merkle tree is a hash tree in which every leaf node is the hash of a data
block and every internal node is the hash of its two children. The single root
hash commits to the entire set of leaves: changing any leaf changes the root.
Merkle trees enable efficient and secure verification of large data structures,
and support compact inclusion proofs of logarithmic size. They are widely used
in version control, peer to peer networks, and blockchains for integrity.
