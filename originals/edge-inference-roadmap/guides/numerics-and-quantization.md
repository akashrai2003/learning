# Numerical correctness, caching and quantization

## A practical parity ladder

Compare scalar operations first, then a block, then the full prefix, then cached decode, then generated behavior. Use a fixed input and capture named intermediates. When a comparison fails, find the earliest divergent tensor; debugging the final paragraph alone gives very little information.

For each tensor record shape, dtype, finite-value status, maximum absolute error, relative error with a safe near-zero denominator, and selected values. No single tolerance applies to every operation or precision. Specify an absolute-plus-relative comparison and justify thresholds using scale, accumulation and dtype.

Argmax equality is insufficient because badly wrong logits can retain the same winner. Argmax disagreement is not always a large error because near-tied values can exchange rank. Quantization adds deliberate approximation, so maintain both numeric and task-level criteria.

## Cache correctness

For each position t, compare uncached execution of tokens 0…t against cached execution up to the same token. Test an empty or smallest valid prefix, several positions, the final capacity boundary, reset and independent sessions. Wrong RoPE position, wrong K/V head indexing and accidental cache reuse are common conceptual errors.

For conventional K/V storage with matching head dimensions, an estimate is:

`KV bytes = 2 × layers × batch × cached_tokens × kv_heads × head_dim × bytes_per_element`.

This is a storage estimate for the stated layout, not a universal formula for every architecture or runtime. Quantized caches add scale/block overhead; sliding windows, shared caches and other attention designs can change allocation. Compare your actual allocation rather than forcing it to fit a simplified equation.

An illustrative configuration of 24 layers, batch one, 4,096 tokens, eight KV heads, head dimension 128 and two-byte elements needs 402,653,184 bytes, or 384 MiB, for K and V. It is not a measured size for a recommended model.

## Quantization definitions

For a chosen symmetric integer range, define a nonzero scale s, quantize using `q = clamp(round(x/s))`, and reconstruct using `x_hat = s*q`. Document whether the negative range is symmetric around zero or uses the extra two's-complement value. Define a safe all-zero-group rule and a precise rounding convention.

Per-channel scales usually follow an output dimension; do not assume row-major storage alone tells you the correct semantic channel. Grouped quantization adds scales and possibly other metadata per group. Compute total storage rather than claiming exactly four bits per parameter from the packed payload alone.

## Two independent errors

First compare original float weights with dequantized weights: this is quantization error. Then compare your optimized quantized kernel against a scalar calculation using those same reconstructed weights: this is implementation/numerical error. Keeping them separate prevents a broken kernel from being excused as expected quantization loss.

## Required edge cases

- All-zero groups and one very large outlier.
- Minimum and maximum codes, half-step rounding and clipping.
- Odd numbers of packed elements and partial final groups.
- Non-square matrices and small dimensions below a vector width.
- Scale indexing and endian/packing agreement between Python and C++.
- Large reductions, finite-value checks and accumulator overflow bounds.

## Calibration and quality

Use representative public inputs to estimate activation behavior. Separate calibration from evaluation. An OCR/document workload can have numbers, identifiers, formatting and long passages unlike casual chat; include relevant examples in the task evaluation.

Measure correctness and citation support for your application, not only generic language-model scores. Report sensitive layers retained in higher precision. A model that becomes smaller yet slower is a valid result when packing, unpacking or unsupported hardware paths add overhead.

## Operations worth understanding deeply

Stable softmax subtracts a row maximum before exponentiation, but an all-masked row still needs a defined policy. RMSNorm's epsilon and accumulation dtype matter. RoPE pairing conventions and scaling must match the model. Attention masking and position indexing must be validated independently of final text quality.

Never enable aggressive floating-point compiler settings or replace stable operations merely because a benchmark improves. Re-run numerical and task evaluation after such changes and document the changed assumptions.
