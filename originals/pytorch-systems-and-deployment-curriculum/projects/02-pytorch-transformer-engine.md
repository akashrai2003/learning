# Project 02 — PyTorch Transformer Engine

**Window:** Weeks 10–15 · **Prerequisites:** Modules 11–24 and Inference 06–12

Implement and train a compact decoder-only Transformer in idiomatic PyTorch,
then evolve it into a controlled inference laboratory. The target is not model
quality at scale. The target is a transparent reference with enough real
structure to expose state, precision, attention, caching, compilation, and
quantization questions.

## Required scope

- Token embedding, positional mechanism (preferably RoPE), normalization,
  residual wiring, causal attention, MLP, logits, loss, optimizer state, and
  checkpointing.
- A shape-aware attention test suite including masks, batch/sequence variation,
  numerical stability, and output parity with an intentionally simple oracle.
- A decode path with a clear KV-cache layout; compare uncached and cached
  generation while separating first-token/prefill from steady-state decode.
- Eager and `torch.compile` variants with graph count, guard/recompile evidence,
  correctness tolerances, and timing methodology.
- One fine-tuning or adaptation experiment via a small torchtune-compatible
  component/recipe, if model format and hardware make it reasonable.
- One low-bit or AMP experiment, with task/sequence-output parity and no claim
  that a weight reduction automatically accelerates end-to-end generation.

## Deliverables

Architecture dossier, state-dict schema, training/restart instructions,
attention/KV-cache diagrams, profiler traces, benchmark matrix, and a source
walk comparing the reference to a production model implementation.

## Mastery gate

Starting from a failed output-parity test, isolate whether the fault comes from
mask construction, layout, cache indexing, dtype/accumulation, model state, or
compiler behavior. Defend the diagnosis with a minimal test—not only a larger
run.

**Inference bridge:** This is the executable counterpart to Inference 06–12 and
29–34. Keep it as the oracle before adopting paged attention, specialized
kernels, or a production serving runtime.
