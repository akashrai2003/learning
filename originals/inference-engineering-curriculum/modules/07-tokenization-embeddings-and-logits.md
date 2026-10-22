# Module 07 — Tokenization, Embeddings, and Logits

> **Phase:** 2 — Architecture Core  
> **Time:** Week 4; two 3-hour sessions plus a 2–4 hour tokenizer parity lab  
> **Prerequisites:** Module 06, Unicode basics, and categorical probability

## Why this matters

The neural network never sees text. It sees IDs produced by an exact tokenizer and template, then returns unnormalized scores over that same vocabulary. A one-token discrepancy changes every subsequent cache position. Many “model quality” failures are actually normalization, special-token, chat-template, vocabulary, or logits-processing mismatches.

## Outcomes

You can:

- explain normalization, pre-tokenization, subword modeling, post-processing, and decoding;
- test tokenization as a reversible, versioned model component;
- derive embedding lookup and vocabulary projection shapes;
- implement stable softmax, greedy selection, temperature, and top-k filtering;
- diagnose tokenizer/template and LM-head mismatches independently of the decoder.

## Mental model: two coupled codebooks

Text crosses a tokenizer pipeline:

`UTF-8 → normalize → pre-tokenize → subword model → special-token post-process → IDs`.

Decoding maps IDs back through token pieces and a decoder; it is not always a simple dictionary join. Byte fallback, leading-space markers, normalization, and invalid byte sequences matter.

The embedding table \(E:[V,D]\) maps discrete IDs into the residual space. The LM head maps the final residual back to \(V\) logits. In a weight-tied model, the same parameters serve both directions, but gather and matrix multiplication are different operations.

## Theory, algorithms, and shapes

Byte-pair encoding starts from smaller symbols and greedily applies learned merges. Unigram tokenization instead selects a segmentation under a learned token probability model. SentencePiece can train directly on raw text and uses an explicit whitespace representation, avoiding a language-specific word splitter.

For IDs \(I:[B,S]\) and table \(E:[V,D]\):

\[
X_{bsd}=E_{I_{bs},d}.
\]

After the decoder, last-position hidden states \(h:[B,D]\) produce:

\[
z=hW^\top+b,\quad W:[V,D],\quad z:[B,V].
\]

Logits are not probabilities. Stable softmax subtracts each row maximum:

\[
p_i=\frac{\exp(z_i-m)}{\sum_j\exp(z_j-m)},\qquad m=\max_j z_j.
\]

Temperature uses \(z/T\), with \(T>0\). Top-k filtering sets all but the largest \(k\) eligible logits to \(-\infty\) before normalization. Greedy decoding uses `argmax` and does not require materializing softmax.

Tokenizer throughput is often measured in input bytes/s, while model work is tokens/s. The ratio tokens/byte changes with language, code, Unicode, and normalization, so text distributions must accompany throughput claims.

## Implementation ladder

1. Implement a reversible byte tokenizer in `reference/tokenizer.py` with explicit BOS/EOS/PAD IDs. It is a systems oracle, not a quality tokenizer.
2. Define `TokenizerSpec` containing vocabulary hash, normalizer behavior, special tokens, added tokens, and template revision.
3. Implement `reference/embedding.py` with bounds-checked gather and optional tied LM head.
4. Implement `reference/sampling.py` with stable softmax, greedy, temperature, and top-k. Accept an explicit RNG generator.
5. Create `tests/tokenizer_cases.jsonl` containing ASCII, leading/trailing spaces, combining characters, multiple scripts, emoji sequences, code, NUL, newline, and literal special-token strings.
6. Add parity tests against a pinned production tokenizer. Compare IDs, decoded bytes/text, offsets where available, and special-token behavior.
7. Add logits tests for extreme values, ties, all-filtered rows, NaNs, and deterministic seeded sampling.

## Two 3-hour sessions

**Session A — text to IDs to logits**

- **0:00–0:35 — Pipeline anatomy:** inspect normalization, pieces, IDs, offsets, and decode for adversarial strings.
- **0:35–1:05 — Subword lab:** perform several BPE merges by hand and contrast a unigram segmentation.
- **1:05–1:15 — Break.**
- **1:15–1:55 — Boundary lab:** implement byte tokenizer, embedding gather, tied/untied projection.
- **1:55–2:30 — Logits lab:** implement stable filtering and selection with explicit edge-case policy.
- **2:30–3:00 — Parity lab:** run the corpus through your pinned production tokenizer and classify differences.

**Session B — boundary forensics**

- **0:00–0:45:** expand the multilingual, code, whitespace, and special-token corpus.
- **0:45–1:30:** compare template rendering, offsets, IDs, and decoded bytes.
- **1:30–2:15:** test tied embeddings and extreme-logit/filtering cases.
- **2:15–3:00:** benchmark text classes and write a mismatch diagnostic decision tree.

## Experiments and deliverables

Submit the tokenizer/spec, embedding and sampling references, parity corpus, tests, and `notes/text-boundary.md`. Measure tokens per UTF-8 byte and tokenization latency across prose, code, and three writing systems. Report distributions, not one English prompt.

Apply two chat templates to the same conversation and show the resulting ID prefixes. Feed both into the same model only after predicting why their outputs may diverge. For tied weights, verify pointer/storage sharing as well as equal values.

## Failure modes

- Assuming `decode(encode(text)) == text` after destructive normalization.
- Adding BOS/EOS both in the template and tokenizer post-processor.
- Treating PAD as EOS or permitting special-token text unintentionally.
- Loading model weights with a tokenizer whose vocabulary size merely happens to match.
- Applying temperature after softmax rather than to logits.
- Computing `exp(logit)` without max subtraction.
- Producing NaNs when every token is filtered.
- Timing Python string conversion as if it were GPU model latency.

## Mastery gate

Given a mismatched generation, isolate raw text, rendered template, IDs, embedding row, logits, filtering, selection, and decoded bytes. The pinned parity corpus must pass; stable softmax must handle logits near dtype limits; seeded sampling must reproduce within the declared environment.

## Primary sources

- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
- [SentencePiece](https://arxiv.org/abs/1808.06226)
- [SentencePiece canonical repository](https://github.com/google/sentencepiece)
- [Hugging Face Tokenizers pipeline](https://huggingface.co/docs/tokenizers/main/pipeline)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- [PyTorch Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)

## Code paths to read

- Tokenizers: `tokenizers/src/tokenizer/`
- SentencePiece: `src/sentencepiece_processor.cc`
- Transformers: `src/transformers/generation/logits_process.py`
- Transformers: `src/transformers/generation/utils.py`

## OSS bridge

Export one production tokenizer artifact and hash every constituent file. Compare slow and fast implementations where both exist. Then trace one generation step in Transformers or vLLM from token IDs to sampled ID, locating template application, selected-token indices, logits processors, and detokenization.

## Next

Module 08 makes the decoder’s residual path numerically exact by implementing normalization placement, epsilon policy, and fused residual semantics.
