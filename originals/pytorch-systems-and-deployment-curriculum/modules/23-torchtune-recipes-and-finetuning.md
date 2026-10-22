# Module 23 — Torchtune Recipes and Fine-Tuning

**Phase:** 4 — PyTorch model engineering and optimization  
**Time:** 9 focused hours  
**Prerequisites:** PyTorch modules/autograd/data loading, Module 22, and basic Transformer architecture.

## Existing inference curriculum bridge

Connect [06 — Transformer forward pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md), [09 — FFN, activations, and SwiGLU](../../inference-engineering-curriculum/modules/09-ffn-activations-and-swiglu.md), [15 — Multimodal inference](../../inference-engineering-curriculum/modules/15-multimodal-inference.md), [18 — Quantization fundamentals](../../inference-engineering-curriculum/modules/18-quantization-fundamentals.md), and [17 — Modern model architecture forensics](../../inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md).

## Why this matters and outcomes

torchtune is a PyTorch-native post-training project, not a magical fine-tuning abstraction. Its real educational value is that recipes expose model construction, tokenizer, dataset, optimizer, checkpointing, distributed wrapping, and loss as ordinary inspectable components. Use it to learn the contracts that must survive later export and serving.

You will be able to run a pinned recipe, trace every object it builds, write a small recipe component, compare full fine-tuning with LoRA/QLoRA, resume exactly, and evaluate both quality and deployability. Its APIs evolve quickly; treat each recipe as source code tied to a torchtune commit and a supported PyTorch version, not an eternal tutorial.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P23 — Torchtune recipes and finetuning**.

1. [Torchtune documentation](https://pytorch.org/torchtune/stable/) — learn the API or system contract.
2. [Torchtune GitHub](https://github.com/pytorch/torchtune) — study the underlying design, paper, or specification.
3. [LoRA paper](https://arxiv.org/abs/2106.09685) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [TorchAO source](https://github.com/pytorch/ao) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Torchtune source](https://github.com/pytorch/torchtune) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Supervised instruction tuning minimizes masked next-token cross entropy:

\[
L=-\frac{1}{\sum_t m_t}\sum_t m_t\log p_\theta(y_t\mid y_{<t},x),
\]

where the loss mask \(m_t\) prevents prompt/padding positions from training the answer model accidentally. Shape discipline is central: tokens `(B,T)`, logits `(B,T,V)`, labels `(B,T)`, attention mask `(B,T)` or an architecture-specific representation. A one-token shift error can yield apparently decreasing loss while training the wrong objective.

LoRA freezes \(W_0\in\mathbb{R}^{d_{out}\times d_{in}}\) and trains a low-rank update \(\Delta W=\alpha BA/r\), with \(A\in\mathbb{R}^{r\times d_{in}}\), \(B\in\mathbb{R}^{d_{out}\times r}\). It lowers optimizer and gradient memory, but target-module choice and merge/export semantics matter. QLoRA combines quantized frozen base weights with trainable adapters; its memory advantage is not evidence that the eventual inference kernel understands the chosen format. Keep checkpoint format, quantization metadata, tokenizer revision, prompt template, and evaluation set in the experiment record.

Recipe configuration should be read as a dependency graph: component factories, YAML/CLI overrides, and a launcher produce a training program. Start by rendering resolved configuration and tracing it to Python symbols. Never change a rank, learning rate, packing mode, and dataset template in one run; that destroys the attribution needed for engineering judgment.

## Implementation ladder in `torch-lab/`

1. **Reference:** implement `torch-lab/finetune/reference_sft.py` for a toy decoder and JSONL instruction dataset. Explicitly build the shifted labels, loss mask, collator, optimizer, and checkpoint manifest.
2. **Optimized:** reproduce it using a pinned torchtune recipe in `torch-lab/finetune/torchtune/`. Record `git rev-parse HEAD`, package lock, resolved config, model/tokenizer revisions, and exact `tune run` command. Add LoRA to one projection family and prove only intended parameters receive gradients.
3. **Integrated:** fine-tune a locally feasible model, export/merge adapters into a clean inference checkpoint, then run the same fixed prompts through the inference curriculum’s reference decoder and vLLM/llama.cpp-compatible path where applicable. Report quality, memory, tokens/s, and format limitations separately.

## Three focused sessions

1. **Recipe archaeology:** clone the pinned source, render config, and make a diagram from CLI override to instantiated dataset/model/loss/checkpointer. Read one model and one recipe end-to-end.
2. **Controlled adaptation:** run full/frozen/LoRA experiments on a small corpus, inspect parameter counts and optimizer states, and verify loss-mask/token alignment with one hand-worked example.
3. **Lifecycle:** interrupt at a known step, resume, compare RNG/checkpoint state, merge adapters, and run a held-out generation plus a deployment smoke test.

## Experiments and deliverables

Create `run_manifest.yaml`, `resolved_config.yaml`, `dataset_card.md`, `adapter_audit.py`, `resume_parity.md`, and `quality_vs_cost.csv`. Include licenses and provenance for weights/data, train/eval split hashes, token counts, seed, package revisions, parameter/optimizer byte counts, loss masking tests, and exact adapter merge procedure.

## Common failure modes

- Fine-tuning an instruction set with labels copied from unshifted tokens.
- Treating a chat template/tokenizer change as harmless preprocessing.
- Resuming weights but not optimizer, scheduler, RNG, or data position.
- Calling every adapter “LoRA” without verifying the targeted module names and scaling.
- Assuming a training-time quantized representation can be consumed by the production inference engine.

## Mastery gate

Starting from a torchtune recipe, explain every trainable tensor, every loss-bearing token, and every persistent checkpoint field. Add one small, tested component or recipe override, reproduce an interrupted run within documented tolerance, and demonstrate a merged adapter’s output parity on fixed prompts.

## Primary sources, code paths, and OSS bridge

- [torchtune repository and recipes](https://github.com/pytorch/torchtune)
- [torchtune documentation](https://pytorch.org/torchtune/stable/)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PyTorch checkpointing note](https://docs.pytorch.org/docs/stable/checkpoint.html)

Trace `recipes/`, `recipes/configs/`, `torchtune/training/`, `torchtune/modules/`, `torchtune/datasets/`, and tests in the pinned checkout. High-value contributions are a minimal resume bug reproduction, a loss-mask test, a model/config documentation correction, or a clearly benchmarked recipe change. Read contribution rules and run formatting/tests before touching a performance path.

## Next

Module 24 makes the training-to-serving quantization boundary explicit with torchao.
