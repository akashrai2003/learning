# Module 24 — Low-Bit Formats: NF4, Codebooks, and GGUF

**Phase:** 4 — Quantization  
**Pacing:** Week 12, paired with Module 23; three 3-hour sessions plus a weekend format audit  
**Prerequisites:** Modules 18–23; scalar quantizers, grouped packing, evaluation, and architecture manifests

## Why this matters

Four bits can encode a uniform integer, a tiny floating-point number, or an index into an arbitrary codebook. These representations have different error, arithmetic, and kernel behavior. GGUF adds another distinction: it is a self-describing container that can hold many tensor encodings, not a quantization algorithm. Engineers must separate numerical format, block layout, file metadata, and executable kernel.

## Outcomes

You can:

- implement nearest-codebook quantization and NF4;
- explain why NF4 is not IEEE-like FP4;
- calculate block-scale and double-quantization overhead;
- parse GGUF metadata and tensor descriptors safely;
- map a GGML quant type to its exact block struct and kernel.

## Mental model: four bits select one of sixteen reconstruction values

For codebook \(C=\{c_0,\ldots,c_{15}\}\) and block scale \(a>0\),

\[
q_i=\arg\min_{j\in[0,15]}|x_i/a-c_j|,
\qquad
\hat x_i=a\,c_{q_i}.
\]

Uniform INT4 uses evenly spaced \(C\). NF4 chooses levels based on quantiles of a standard normal distribution, motivated by approximately normally distributed pretrained weights. A practical implementation normalizes a block by its absolute maximum, stores 4-bit indices, and stores the scale. NF4 has no sign/exponent/mantissa interpretation; it is a lookup-table code.

QLoRA also introduces **double quantization**: quantize the block scales themselves to reduce metadata. If a block has \(B\) values and its scale costs \(b_s\) bits, raw effective cost is

\[
4+\frac{b_s}{B}\quad\text{bits/weight},
\]

before double-quantization metadata, padding, and alignment. Reconstruct scales in the correct higher precision; codebook lookup and nested scale decoding affect kernel design.

Codebooks adapt well to one distribution but can fail under skew, heavy tails, or post-training transformations. Compare expected error empirically rather than treating “NF4” as universally superior.

## GGUF is a container

GGUF stores a versioned header, typed key/value metadata, tensor descriptors, aligned tensor data, architecture/tokenizer information, and per-tensor `ggml_type`. Files may contain F32, F16, Q8, several Q4/Q5/Q6 K-quants, IQ variants, or MXFP4 in different tensors. A filename like `Q4_K_M` commonly describes a mixed policy, not one universal four-bit layout.

Each GGML type has a block struct. A classic Q4_0 block represents 32 logical values with one FP16 scale and 16 packed bytes, an ideal

\[
\frac{16+16\cdot8}{32}=4.5\text{ bits/value}.
\]

That calculation does not describe Q4_K, IQ4, or another type. Inspect `ggml-quants.h` for the exact commit, then follow its quantize/dequantize and dot-product implementation. Block size, scale/min fields, nibble arrangement, and SIMD/GPU packing are ABI.

## Implementation ladder

### 1. Reference

Add `inference-lab/reference/codebook_quant.py` with uniform INT4 and NF4 codebooks, nearest-level selection, block absmax scaling, packing, and dequantization. Generate the NF4 table from the documented rule or use a golden canonical table; test every index.

### 2. Optimized

Vectorize lookup/packing and implement a fused block dot product that consumes packed indices and scales without writing a full dequantized tensor. Compare Gaussian, Laplace, Student-t, and transformed real weights.

### 3. Integrated

Write a read-only GGUF inspector under `engine/formats/gguf_inspect.py`: validate magic/version, bounds, alignment, metadata types, tensor offsets, and quantization version. Convert a small pinned model using the official tool, run it in `llama.cpp`, and map one hot tensor from file bytes to its dot kernel.

## Three 3-hour sessions

1. **Codebook theory:** derive nearest-code assignment, implement NF4/uniform INT4, and compare error across distributions.
2. **Block and container:** calculate true bits/value, implement golden block packing, and parse a GGUF header/tensor directory.
3. **Runtime source walk:** trace `convert_hf_to_gguf.py` → `llama-quantize` → GGUF loader → `ggml` type traits → CPU/GPU dot kernel.

The weekend audit reports format version, tensor-type histogram, metadata overhead, file versus resident memory, quality, and prefill/decode performance.

## Experiments and deliverables

Commit:

- `reference/codebook_quant.py`;
- `tests/test_codebooks_and_gguf.py`;
- `engine/formats/gguf_inspect.py`;
- `benchmarks/results/codebook-distributions.json`;
- `notes/source_walks/gguf-q4-block.md`.

Keep golden byte fixtures tiny and versioned. Fuzz malformed lengths and offsets without allocating based on untrusted counts. For model quality, compare identical tokenizer metadata and prompt formatting.

## Common failure modes

- Calling NF4 a 4-bit floating-point format.
- Assuming normally distributed weights remain normal after scaling or fine-tuning.
- Omitting scales and nested-scale metadata from memory claims.
- Treating GGUF, GGML type, and a filename quant label as synonyms.
- Applying one Q4 block formula to every Q4 variant.
- Loading malformed metadata or tensor offsets without bounds checks.
- Comparing a CPU GGUF runtime with a GPU framework and attributing all differences to quantization.

## Mastery gate

Implement bit-exact NF4 block quantization, calculate effective bits/value including scale storage, parse a GGUF tensor descriptor, and identify the exact source struct and dot kernel for its `ggml_type`. Explain a distribution where uniform INT4 beats NF4.

## Canonical code paths and OSS bridge

Read the QLoRA paper with bitsandbytes’ canonical [NF4 table and 4-bit functions](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py). For GGUF, use the [official specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), `llama.cpp`’s [`tools/quantize`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md), `ggml/src/ggml-quants.c`, and `ggml/include/ggml.h`/quant block headers. Strong OSS tasks include malformed-file tests, metadata docs, golden block vectors, or a benchmark for a missing architecture/shape.

## Primary and authoritative sources

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [bitsandbytes canonical quantization implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp quantization tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

## Next

Module 25 moves from integer grids and codebooks to native FP8 formats, tensor scaling, accumulation, and training/inference recipes.
