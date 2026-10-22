# Android and accelerator deployment guide

## The layered model

Your application owns user interactions and lifecycle. A native interface owns sessions and callbacks. The runtime owns execution and cache operations. A backend maps supported work onto CPU, GPU or NPU. The operating system and OEM drivers control the hardware access available to all of these layers.

A model that runs on CPU does not automatically run unchanged on the NPU. Operator coverage, graph shape, quantization, artifact format and driver compatibility can differ. A working example for a reference device does not prove support for every retail phone with the same SoC.

## Route A: llama.cpp

Use the CPU route first. It offers a practical baseline and source code you can trace from CLI to model evaluation. The current upstream Snapdragon guide describes CPU, Adreno OpenCL and experimental Hexagon execution. Follow the pinned checkout's setup and inspect its backend capabilities and logs. Generic Android Vulkan instructions are a separate possible GPU route; do not mix flags or prerequisites from different backends.

Keep a small known-working model and a very short prompt for initialization tests. Confirm actual backend execution before measuring long prompts. Different quant formats can have different support and kernel performance. A backend may initialize yet fall back for important operators.

## Route B: ExecuTorch

First export a tiny linear model and establish Python-to-runtime output agreement. Then learn shape constraints, operator decomposition, memory planning and delegation. Only after this should an LLM recipe become the target.

Use matching exporter, runtime, quantization and backend versions. Classify failures by stage: Python capture; graph transformation; serialization; target preparation; runtime loading; operator execution; numerical output. This classification makes an issue much easier to diagnose than 'NPU does not work.'

## Route C: Qualcomm Genie or Qualcomm backend

Choose one official recipe and one supported model. The Genie tutorial names Snapdragon 8 Elite Gen 5 among Android targets, but actual operation depends on compatible model assets, SDK/runtime and device support. Record all relevant versions, not only the chipset name.

Begin with supplied or officially generated assets. Your own GGUF or educational checkpoint should not be assumed compatible. Conversion can require a different graph and quantization workflow. Some steps require host resources or account access; read those requirements before downloading a large model.

If blocked, save a minimal reproduction with exact versions, first complete error and a small input. Continue the curriculum with CPU or another verified backend. Do not root the phone or alter firmware simply to satisfy this curriculum.

## JNI contract checklist

- Who owns the engine and who owns each session?
- Which thread runs generation and which thread receives callbacks?
- Can a callback re-enter the engine?
- What happens if cancel races with completion?
- What happens if destroy is requested during generation?
- How are native failures translated into Kotlin results?
- How are local/global JNI references released?
- How are partial UTF-8 token bytes assembled into valid text?
- Which state survives activity recreation and which is rebuilt after process death?

Use a simple serialized worker before adding parallel requests. Avoid holding a native mutex while invoking external callbacks that could re-enter the runtime. Preserve debug symbols for diagnosing crashes.

## Android lifecycle policy

Do not confuse an activity's existence with the native engine's lifetime. Decide explicitly whether background generation is cancelled or handled using an Android-supported execution model. Avoid promising background survival without implementing the required lifecycle behavior.

Use conservative model and context budgets. The phone's total RAM is shared, and the OS can reclaim processes. A graceful rejection is preferable to allocating until the process disappears. Your capstone should explain this policy to the user in ordinary terms.

## Profiling limits

Use Simpleperf and Perfetto in modes supported by your device and app build. Some counters or system-wide observations may be restricted. Record what is visible and use stage instrumentation where needed. Do not infer precise NPU utilization, energy or transfer cost when the tools do not expose it.

## Version diary

For each working configuration, save phone model/firmware, Android version, ABI, NDK/compiler, runtime commit, SDK version, backend, model revision, tokenizer/template, quantization, shapes and exact launch settings. After an upgrade, rerun the small known-working case before a full benchmark.
