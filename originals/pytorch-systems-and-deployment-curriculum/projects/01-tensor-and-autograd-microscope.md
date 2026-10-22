# Project 01 — Tensor and Autograd Microscope

**Window:** Weeks 4–11 · **Prerequisites:** Modules 01–10

Build a small executable laboratory that makes PyTorch tensor semantics visible
and falsifiable. This is not a NumPy tutorial. Its value is the ability to
predict what PyTorch will do before execution and to distinguish storage,
metadata, device scheduling, and differentiation behavior.

## Required experiments

- Storage offsets, sizes, strides, views, reshapes, transposes, contiguity, and
  aliasing; include both safe and unsafe-looking cases with assertions.
- Broadcasting, advanced indexing, `einsum`, and in-place mutation, each with a
  shape table and a property test.
- CPU/CUDA transfers, pinned-memory input staging, stream ordering, and an
  explicit synchronization experiment. State what timing is invalid without a
  synchronize boundary.
- A scalar or tiny-tensor reverse-mode autograd reference, then compare it to
  `torch.autograd.grad`, `gradcheck`, `no_grad`, and `inference_mode` behavior.
- Deliberate errors: a version-counter/in-place-autograd failure, a non-leaf
  gradient surprise, a dtype accumulation error, and a view/contiguity failure.

## Deliverables

1. A `pytest` suite organized by semantic contract, not just API name.
2. A one-page storage/alias model, an autograd graph diagram, and a table of
   operations that return views versus materialized tensors for the cases you
   tested.
3. A short profiler trace contrasting CPU time, GPU time, and asynchronous
   enqueue time.
4. A source-walk note for one relevant PyTorch test or implementation path.

## Mastery gate

Given an unfamiliar short tensor program, annotate shape, stride, aliasing,
leaf/gradient status, and expected synchronization behavior by hand; then write
a test that checks the prediction. Explain one case where `reshape` does not
have the same storage consequence as the mental shortcut “it is just a view.”

**Inference bridge:** Inference modules 03, 04, and 36. This project is the
foundation for understanding KV-cache layout, attention transposes, and kernel
memory access later.
