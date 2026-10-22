# Module 13 — Data Pipelines, Datasets, and DataLoaders

> **Phase:** I — Engineering foundations  
> **Time:** 9–11 focused hours across three sessions  
> **Prerequisites:** Modules 08–12 and basic Python multiprocessing awareness

## Existing inference curriculum bridge

Connect this work to [Module 07 — Tokenization, Embeddings, and Logits](../../inference-engineering-curriculum/modules/07-tokenization-embeddings-and-logits.md), [Module 15 — Multimodal Inference](../../inference-engineering-curriculum/modules/15-multimodal-inference.md), [Module 34 — Continuous Batching, Chunked Prefill, and Prefix Cache](../../inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md), and [Module 44 — Profiling, Benchmarking, and Correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Training batches and serving batches solve different scheduling problems, but both need clear ownership, layout, and latency contracts.

## Why this matters and outcomes

Data loading is part of the model’s effective systems design. A GPU can idle behind tokenization, image decode, padding, collation, worker startup, host copies, or a duplicate sample stream. A fast but semantically wrong pipeline can invalidate training just as completely as a wrong optimizer.

By the end, you can choose map-style versus iterable datasets; design a sample and batch contract; control sampling, sharding, collation, padding, and worker state; use pinned memory as an evidence-backed transfer optimization; and diagnose data-loader stalls, duplication, and multiprocessing hazards with a reproducible benchmark.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P13 — Data pipelines, datasets, and dataloaders**.

1. [Data loading tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) — learn the API or system contract.
2. [torch.utils.data](https://docs.pytorch.org/docs/stable/data.html) — study the underlying design, paper, or specification.
3. [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch examples](https://github.com/pytorch/examples) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch recipes](https://docs.pytorch.org/tutorials/recipes/recipes_index.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

A map-style Dataset implements indexed access and usually a stable length. It works naturally with samplers that choose or reorder indices. An IterableDataset yields a stream and may not have random access or finite length; it is appropriate for streaming shards, generated data, or stateful sources. These are not interchangeable interfaces. In multi-worker iterable loading, each worker receives its own dataset instance, so it must shard the source explicitly using worker information or it may emit duplicate records.

A DataLoader is a pipeline coordinator: sampler or iterable source, optional batching, collation, worker execution, memory pinning, and delivery to the training loop. The default collate function stacks compatible samples. Real language and multimodal examples often require a custom collate function to tokenize, pad to a per-batch maximum, build attention masks, preserve labels, and retain metadata. The batch contract should state which fields are Tensor, list, string, CPU-only metadata, padding convention, dtype, and expected device. Do not move data to CUDA inside dataset workers; workers should produce CPU data and the main training process should control the device transfer.

Sampling changes the data distribution seen by optimization. Shuffle is not the same as replacement sampling, weighted sampling, or distributed sharding. In distributed training, every rank must have an intentional sample partition and epoch-dependent ordering. The later randomness module will make this reproducible; here, assert IDs and count examples so an accidental duplicate or missing tail is visible.

With num_workers=0, loading happens in the main process and is the best debugging baseline. With workers, process-start behavior, serialization/pickling, global state, file handles, and worker seeding matter. Keep Dataset initialization lightweight and use worker_init_fn or get_worker_info only with a documented state plan. On CUDA platforms, initializing CUDA before forked workers is hazardous; use the documented spawn/forkserver guidance when CUDA work and multiprocessing intersect. persistent_workers can amortize startup, but it also makes stale worker state a real lifecycle concern.

Pinned memory prepares CPU buffers for faster DMA, as introduced in Module 09. It improves a pipeline only if the transfer path and workload justify it. Tune batch size, workers, prefetching, collation cost, pinning, and device transfer separately. Measure loader wait time, CPU utilization, storage throughput, and GPU utilization; avoid adopting a popular worker count without evidence from your machine and dataset.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/data/token_dataset.py with a deterministic map-style variable-length token dataset. Return a typed sample dictionary with input_ids, labels, length, and sample_id.
2. Implement collate_causal_lm: pad per batch, form attention mask and shifted labels, enforce int64 token IDs, and test padding/ignore-index behavior on hand-built examples.

**Optimized**

3. Create an IterableDataset that reads synthetic shards. Use get_worker_info to shard records and test that one and multiple workers produce neither duplicates nor omissions.
4. Benchmark num_workers, persistent_workers, pin_memory, batch size, and prefetch policy independently. Record data wait versus device compute rather than only epoch time.

**Integrated**

5. Add fault tests for a corrupted sample, a slow worker, an unpicklable closure, duplicate shard assignment, and a data source that raises after N samples.

## Three focused sessions

1. **Contracts and collation (3h):** define sample/batch dataclasses or typed dictionaries, write variable-length causal-LM collation, and prove masks/labels with literal examples.
2. **Workers and sharding (3h):** make the intentional duplicate IterableDataset failure, repair it with worker-aware sharding, then test lifecycle and deterministic IDs.
3. **Pipeline evidence (3h):** sweep one variable at a time, profile loader waits and GPU utilization, and produce a recommendation tied to the target hardware and data source.

## Experiments and deliverables

Commit datasets, collate functions, worker/sharding tests, a data-contract document, and a benchmark report. The report must include sample count, duplicate/missing checks, padding ratio, worker configuration, device-transfer policy, throughput, latency distribution, and environment. Keep a small CPU-only test path to debug correctness without CUDA or workers.

## Common failure modes

- Moving CUDA tensors or doing GPU initialization inside ordinary DataLoader workers.
- Treating IterableDataset replication across workers as automatic sharding.
- Hiding tokenization, random augmentation, and padding semantics inside an opaque collate function.
- Reporting only examples per second while the GPU is idle or batches contain mostly padding.
- Leaving persistent worker state alive across changed dataset configuration.
- Failing to test the last partial batch and ignore-index semantics.

## Mastery gate

Build a multi-worker variable-length data pipeline that has no duplicate/missing IDs, correct masks/labels, documented padding behavior, and a defensible transfer policy. Given a GPU-idle trace, isolate whether storage, decode/tokenization, collation, worker scheduling, pinning, or model compute is responsible.

## Primary sources and canonical code paths

- [PyTorch data loading API](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)
- [PyTorch torch.multiprocessing](https://docs.pytorch.org/docs/stable/multiprocessing.html)
- Inspect [dataloader.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py), [worker.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/worker.py), [fetch.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/fetch.py), and [sampler.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py).

## OSS bridge and next

Good upstream work includes a minimized worker/sharding regression, a collation edge-case test, or documentation supported by a cross-platform reproducer. A raw performance claim is not enough; attach configuration and profiling evidence. Module 14 turns correctly delivered batches into controlled parameter updates, schedules, and resumable optimization state.
