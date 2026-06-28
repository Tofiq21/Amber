# Banked Compute
Expensive computation can be performed once and stored for cheap reuse. Trained
model weights are a familiar example: training burns enormous compute, and the
frozen weights are reused at inference. Embedding a corpus is another: the heavy
pass runs once, and the resulting vectors are reused for retrieval. The stored
result is portable and works offline, but a small runtime is still needed to use
it. The expensive work is banked; only light work remains at task time.
