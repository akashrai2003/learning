# Benchmarking protocol

## First define the question

'Which backend is fastest?' is underspecified. A useful question is: 'For this model and quantization, with 512 actual prompt tokens and up to 128 generated tokens, which verified backend gives the lowest median warm application latency under these phone conditions?'

Keep distinct questions for startup, prefill, decode, memory, sustained behavior, energy and task quality. A single headline tokens-per-second result cannot answer all of them.

## Timing boundaries

Define an application request start, admission time, tokenization end, model-ready time, prefill start/end, first visible text and final completion. TTFT can mean different things in different tools. State your definition in every report.

Runtime-only prefill may exclude loading and tokenization. Application TTFT includes the stages that actually occur before the first visible output, including queueing. Decode throughput should use an explicit interval and token count; some tools count the first generated token differently. Preserve the tool's semantics rather than silently mixing definitions.

## Controlled variables

Record model and tokenizer revisions, quantization format, prompt template, actual token IDs/counts, context settings, sampling, output limit, runtime commit, build flags, backend, thread count and hardware configuration. A parameter called 'batch size' may describe token processing rather than concurrent users; verify its meaning in the runtime.

Record charging, battery level, screen state, approximate ambient conditions and available thermal status. Randomize or alternate backend order when practical so the last backend is not always measured on the hottest phone. Cool-down policy should be consistent and documented. Do not disable device thermal protections.

## A manageable initial matrix

Start with one small model, two thread counts, three prompt lengths and three repetitions: 18 runs. Use requested token counts as targets, but save actual counts after tokenization. A tiny matrix you inspect is better than thousands of unexplained records.

Then compare one accelerator against CPU for selected representative workloads. Expand repetition counts only after verifying logs and the experiment contract. Three runs help detect gross variation but cannot establish reliable p95 or p99 latency. Save distributions and do not manufacture confidence from small samples.

## Warm and cold terminology

Separate fresh process, unloaded model, prepared graph, resident model and reused session. A process restart does not necessarily clear the filesystem cache. Do not call it a cold-storage benchmark without controlling that variable. Never clear system caches or change privileged settings simply to fit a label.

## Memory measurements

Weight file bytes, mapped virtual bytes, RSS, PSS, heap allocation, accelerator buffers and KV allocation are different quantities. Some views overlap. A mapped file may occupy little resident memory until pages are touched; counting file size plus RSS may double-count.

Use runtime allocation reports and available Android process tools, preserve raw measurements and explain their limits. An app's safe memory budget is below total phone RAM and depends on other processes and OEM behavior. Observe load and generation peaks, not only idle memory.

## Energy and thermal measurements

Power is a rate; energy is accumulated power over time. Battery percentage is coarse and confounded by charging, screen use and background activity. Use a suitable external meter or a trustworthy device measurement path for precise energy claims, and describe what the meter includes.

Without that, report sustained latency and thermal status and label battery-drain observations exploratory. Do not infer precise joules per token from a short run. A lower-power backend may still use more total energy if it runs much longer.

## Comparison fairness

When architecture, quant, template or generation length differs, call it an end-to-end deployment comparison. A supported NPU quant versus a CPU GGUF quant can be useful, but it does not isolate the processor alone. Include quality checks so a fast degraded model does not win by omission.

Synthetic fixed-output-length tests isolate runtime work but differ from normal EOS-terminated application requests. Keep both if useful and label them separately. Never count padding or ignored work as useful generation.

## Minimal report

1. Question and acceptance criterion.
2. Configuration and measurement boundaries.
3. Raw-record location and failure counts.
4. Latency/memory/quality table with sample sizes.
5. Sustained-run behavior and measurement limitations.
6. Hypothesis, interpretation and next experiment.

Keep one successful optimization and one failed hypothesis in your portfolio. The latter demonstrates that you can reject an attractive explanation using evidence.
