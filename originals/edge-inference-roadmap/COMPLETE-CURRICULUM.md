# Complete curriculum reading edition

Use the split files for daily work. This edition combines them for searching and printing. Relative links refer to the original unpacked folder layout.

<!-- Source: README.md -->

# Edge inference engineering: a 24-week daily curriculum

Prepared for Akash Rai • 20 October 2026

From production GenAI application engineering to C++ inference internals, Android deployment, quantization and accelerator evaluation.

## Start here

1. Read [the operating guide](00-operating-guide.md).
2. Check [setup and first commands](01-setup.md).
3. Open [week 01](weeks/week-01.md) and complete Day 001.
4. Record progress in [the checklist](progress.md) and your own copy of [the daily log](templates/daily-log.md).
5. Use [lab contracts](guides/lab-contracts.md) whenever a daily task refers to a larger implementation.

## Scope and time

There are **168 numbered daily sessions across 24 weeks**. Days 1–6 each week have a 150-minute core and an optional 30-minute extension. Day 7 has a 120-minute review, repair and reconstruction session, with up to 60 optional minutes. The core is approximately **408 hours**; it becomes longer when a milestone needs additional work. At six months this is a substantial part-time commitment.

The calendar is a suggested sequence, not a promise of C++ mastery or hardware compatibility. Do not compress an unresolved numerical or lifetime bug into tomorrow's prerequisites. Repeat a session or spread a difficult lab over more days.

Your existing background in GenAI applications, vLLM serving, document intelligence and on-prem deployment is the starting point. Python fundamentals and general RAG tutorials are intentionally not the main course. C++ starts from compilation, variables and functions.

## Contents

- [Operating guide and study method](00-operating-guide.md)
- [Setup, first commands and repository structure](01-setup.md)
- [24-week navigation](02-week-index.md)
- [Daily completion checklist](progress.md)
- [C++ learning map](guides/cpp-learning-map.md)
- [Implementation lab contracts](guides/lab-contracts.md)
- [Benchmarking protocol](guides/benchmarking.md)
- [Numerical correctness and quantization](guides/numerics-and-quantization.md)
- [Android and accelerator deployment](guides/android-and-backends.md)
- [Production engineering and troubleshooting](guides/production-and-debugging.md)
- [Capstone specification and evaluation](guides/capstone.md)
- [Milestones and self-assessment](guides/milestones.md)
- [Optional next eight weeks](guides/next-eight-weeks.md)
- [Reading and source catalogue](resources.md)
- [Templates](templates/daily-log.md)
- `schedule.csv`: filterable daily schedule; open with a spreadsheet or text editor.
- `COMPLETE-CURRICULUM.md`: all curriculum Markdown combined into one searchable reading file. Original relative links refer to the unpacked folder structure.

## What this pack contains

Assignments, explanations, implementation contracts, test cases, references and recording templates. It deliberately does not provide completed inference-engine solutions: writing and debugging those implementations is the learning objective. Example commands are scaffolding and need your actual paths and compatible tool versions.

## Research boundary

Core upstream runtime, export, build and profiling references were checked while preparing this curriculum on 20 October 2026. Additional reference links are curated entry points, not claims that every tutorial or model was executed. The curriculum was structurally validated; no build, benchmark or phone result is presented as already tested on your hardware. Verify current compatibility at the pinned versions you choose.


<!-- Source: 00-operating-guide.md -->

# How to work through the curriculum

## Intended destination

By the end, you should be able to trace a request from tokenization to sampling, explain model and session memory, implement and test a small decoder, profile an Android runtime, and justify a deployment choice. Your strongest portfolio evidence will be an educational native engine, a reproducible device benchmark suite, and a reliable offline document application.

The goal is independent engineering judgment. Completing 168 checkboxes without understanding ownership, numerics or measurement is not success.

## Daily time budget

On ordinary days use **15 minutes recall, 30 minutes study, 75 minutes implementation, 20 minutes verification, 10 minutes logging**. This is 150 minutes. If you have only two hours, reduce reading to 20 and implementation to 55; carry unfinished scope forward. If you have three hours, add a focused extension, not a second unrelated topic.

During recall, close yesterday's code and explain one concept or reconstruct one function. During study, read the named topic from that week's linked resources. During implementation, write your own small solution. During verification, use an independent reference or failure case. During logging, record what changed and what remains uncertain.

On Day 7 use 20 minutes recall, 60 minutes repair or reconstruction, 25 minutes milestone review, and 15 minutes planning. This is a two-hour working review, not an obligatory large new project. If you need a rest day, move it rather than abandoning the sequence.

## Six rules for working independently

1. Before code, write the function contract: shapes, dtypes, ownership, errors and examples.
2. Implement a small scalar reference before vectorization, threading or acceleration.
3. Save the first failing input and locate the first wrong intermediate value.
4. Use reference implementations after your own attempt, then reconstruct the key idea without looking.
5. Keep a stable checkpoint and test fixtures while debugging. Change one source of behavior at a time.
6. Explain every claim in a report using code, a test, a raw record or an explicitly labeled assumption.

## An AI-use protocol

Spend the first focused implementation block without code generation. If blocked, consult the compiler, debugger and official reference first. Then ask for one hint, a conceptual explanation, an additional edge case, or review of your own code. Avoid requesting the finished kernel or module.

Keep an assistance log: problem, what you tried, hint received, and what you later reconstructed unaided. Reimplement one assisted function the next day and one week later. Tool use is not failure; unexplained dependence is the thing to reduce.

Good questions include: 'What lifetime assumption is unsafe here?', 'Which experiment distinguishes allocation overhead from compute?', and 'Give one counterexample to this quantization test.' A less useful request is 'Write my entire runtime.'

## Managing difficult tasks

Some export or NPU steps may consume days because of compatibility. Time-box initial environment diagnosis to one session. Save versions, commands and the first complete error. Attempt a minimal model. If the issue is outside your code, continue with CPU deployment and preserve the accelerator issue as a research branch. Never let unsupported firmware prevent learning inference fundamentals.

A hard task can be divided into: define the interface; implement the simplest case; add one boundary; compare a reference; integrate. Do not count optional stretch exercises as required prerequisites.

## Milestone policy

Pause after weeks 6, 10, 14, 17, 20 and 24. Read the milestone rubric. If correctness or ownership is unresolved, add repair days before dependent work. If performance is disappointing but understood and measured, you may progress. A well-explained slow implementation is a valid baseline.

## What to defer

Defer advanced metaprogramming, lock-free queues, writing a full compiler, training large models, custom GPU kernels for every operator, and simultaneous study of several runtimes. Server continuous batching and paged KV allocation are useful later; they are not prerequisites for batch-one phone inference.

## Repository habits

Use small commits for contracts, reference operations, tests, measurements and integration. Keep checkpoints outside Git; track source, revision, terms and hashes in a manifest. Never commit API keys, vendor credentials or private client documents. A local educational repository does not need to be public to be useful.


<!-- Source: 01-setup.md -->

# Setup and first commands

## Computer and phone roles

Use your computer for editing, building, exporting checkpoints and running Python evaluation. Use the phone as a measured Android target. Start without root. Do not change kernel settings or thermal protections to complete this curriculum.

Record the exact handset model and Android build. The supplied Snapdragon 8 Elite Gen 5 and 16 GB RAM are the planned target, but available RAM, OEM libraries and supported accelerators must be observed. Disable marketing RAM expansion in your capacity reasoning: it is not additional physical memory.

## Host prerequisites

Install a supported C++ compiler (Clang or GCC), debugger, CMake, a build tool such as Ninja, Git, and a Python virtual environment. Later add Android Studio, SDK platform tools and a compatible Android NDK. Use official installers or your operating system's supported packages. Vendor tools may require a particular supported host environment; follow that release's requirements.

For Windows, a native Android Studio environment is sufficient for the Android side. A Linux development environment can simplify host C++ and vendor workflows, but do not assume every USB or vendor-tool step works identically inside WSL. Start with the smallest host build and a working native ADB connection.

## First C++ program

Create `hello.cpp` yourself containing a main function that prints a short line. On a compatible host with Clang, a basic command is:

```bash
clang++ -std=c++20 -Wall -Wextra -Wpedantic -g hello.cpp -o hello
./hello
```

The equivalent GCC command uses `g++`. On Windows use the correct executable name or IDE terminal for your compiler. Explain the language standard, warnings, symbols, input and output arguments. Add conversion warnings later and resolve them deliberately rather than globally suppressing them.

## Minimal CMake scaffold

```cmake
cmake_minimum_required(VERSION 3.20)
project(edge_learning LANGUAGES CXX)
add_executable(hello hello.cpp)
target_compile_features(hello PRIVATE cxx_std_20)
```

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

Single-configuration generators use `CMAKE_BUILD_TYPE`; multi-configuration generators also need the appropriate build configuration. Keep debug, release, sanitizer and Android outputs in separate directories. Use your installed stable CMake guide rather than assuming the newest online tutorial matches it.

## First ADB checks

After installing platform tools, enable developer options and USB debugging on the phone, authorize your computer, then use:

```bash
adb devices
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.product.cpu.abi
adb shell cat /proc/meminfo
```

These inspect the device. Memory output is system-wide and does not establish an app's safe allocation budget. Do not paste private device identifiers into public reports.

## Runtime setup

Follow [upstream Android instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md) for CPU deployment and [the Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md) for the selected accelerator path. Pin a commit and record it. Use the commands from that checkout because binary names, flags and SDK requirements change.

Choose a small official checkpoint and a compatible model artifact. Prefer approximately 1B parameters for runtime baselines and approximately 100–400M for your own educational engine. Do not assume every architecture is supported by every backend or that all quants are accepted. Download model assets manually according to their terms; this pack contains no model weights.

## Suggested learning repository

Create these directories yourself: `cpp-basics`, `tensor-core`, `python-reference`, `native-engine`, `android-app`, `benchmarks`, `evaluation`, `fixtures`, and `notes`. Add a root README explaining what belongs where. Keep large model files and generated build outputs out of Git.

Your benchmark script should use subprocess argument arrays, explicit timeouts, and captured stderr. Do not build a shell command by concatenating untrusted prompts or file paths. Preserve the exact raw output before attempting to parse it.

## Initial decisions to record

- Host operating system, compiler, CMake, Python and Android tooling versions.
- Phone model, Android version, firmware identifier and available storage.
- Runtime commit, backend and build configuration.
- Model source, revision, architecture, tokenizer, chat template, quant and hash.
- Context limit, maximum output tokens and sampling settings.
- Which steps are already verified and which are still assumptions.


<!-- Source: 02-week-index.md -->

# Weekly navigation

| Week | Days | Focus |
|---|---|---|
| [01](weeks/week-01.md) | 001–007 | First C++ programs and an Android baseline |
| [02](weeks/week-02.md) | 008–014 | Control flow, numeric types and reproducible experiments |
| [03](weeks/week-03.md) | 015–021 | References, pointers, containers and debugging |
| [04](weeks/week-04.md) | 022–028 | Classes, RAII, moves and a minimal tensor |
| [05](weeks/week-05.md) | 029–035 | CMake, testing, standard algorithms and checkpoint I/O |
| [06](weeks/week-06.md) | 036–042 | Concurrency, Python bindings and systems foundations |
| [07](weeks/week-07.md) | 043–049 | Python transformer mathematics and reference operations |
| [08](weeks/week-08.md) | 050–056 | Checkpoint parity, tokenization and cached generation |
| [09](weeks/week-09.md) | 057–063 | C++ transformer kernels and checkpoint integration |
| [10](weeks/week-10.md) | 064–070 | A complete educational C++ inference engine |
| [11](weeks/week-11.md) | 071–077 | Profiling and the memory hierarchy |
| [12](weeks/week-12.md) | 078–084 | INT8 quantization and quality evaluation |
| [13](weeks/week-13.md) | 085–091 | INT4 packing, group quantization and fused dequantization |
| [14](weeks/week-14.md) | 092–098 | ARM NEON, threading and measured optimization |
| [15](weeks/week-15.md) | 099–105 | Android NDK deployment and native diagnostics |
| [16](weeks/week-16.md) | 106–112 | Kotlin, JNI and a minimal offline application |
| [17](weeks/week-17.md) | 113–119 | Lifecycle, sustained performance and resource policies |
| [18](weeks/week-18.md) | 120–126 | ExecuTorch export, graph lowering and delegation |
| [19](weeks/week-19.md) | 127–133 | Qualcomm NPU deployment and vendor runtime constraints |
| [20](weeks/week-20.md) | 134–140 | Backend comparison and inference infrastructure |
| [21](weeks/week-21.md) | 141–147 | Offline document assistant: retrieval and resource ownership |
| [22](weeks/week-22.md) | 148–154 | Capstone: memory management, quality and optional multimodality |
| [23](weeks/week-23.md) | 155–161 | Evaluation, regression testing and sustained application behavior |
| [24](weeks/week-24.md) | 162–168 | Release engineering, portfolio and the next specialization |


<!-- Source: weeks/week-01.md -->

# Week 01: First C++ programs and an Android baseline

Days 001–007 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md): Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 001 — Development inventory

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Compiler versus interpreter; source, executable, host and target.

**Build:** Create a learning repository; record phone model, real RAM, Android version, available storage, computer compiler and Python versions; compile a hello-world C++ program.

**Verify:** Run the executable from a terminal and explain every command; distinguish physical RAM from RAM expansion.

**Save:** code or configuration changes, any raw observations, and `notes/day-001.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 002 — Variables and initialization

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Basic types, initialization, cout, cin, statements and compiler diagnostics.

**Build:** Write a CLI that asks for parameter count and bit width and estimates raw weight bytes; use double for the first version.

**Verify:** Check 1B at 4 bits gives 500 million decimal bytes; distinguish GB from GiB; reject a negative input.

**Save:** code or configuration changes, any raw observations, and `notes/day-002.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 003 — Functions and parameters

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Return types, local scope, pass by value and function declarations.

**Build:** Split the estimator into conversion, estimation and presentation functions; add KV and runtime allowance as explicit user inputs.

**Verify:** Explain why the result is an estimate rather than phone capacity; manually check three configurations.

**Save:** code or configuration changes, any raw observations, and `notes/day-003.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 004 — Phone development connection

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** ADB, developer options, host-target separation; upstream Android build guide.

**Build:** Enable debugging, connect by USB, inspect device properties and run a harmless shell command; choose one small model and record its origin and terms.

**Verify:** Save device metadata and the exact model identity; do not claim an accelerator works before a model runs.

**Save:** code or configuration changes, any raw observations, and `notes/day-004.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 005 — First native baseline

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Runtime commit, release build, model hash, CPU execution and context limits.

**Build:** Follow upstream instructions for the pinned checkout; deploy a CPU executable and run one short prompt; save the complete command and logs.

**Verify:** Confirm generation occurs on the phone and backend logs identify CPU; record actual tokens rather than word counts.

**Save:** code or configuration changes, any raw observations, and `notes/day-005.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 006 — Simple Python experiment runner

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** subprocess argument lists, timeouts, JSON records and error capture.

**Build:** Write an ADB wrapper for one inference run; capture stdout, stderr, return code, elapsed time and configuration.

**Verify:** Disconnect the device or use a bad path and confirm a structured failure record; do not hide errors.

**Save:** code or configuration changes, any raw observations, and `notes/day-006.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 007 — Week-one reconstruction

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Recall variables, functions and the host-target build path.

**Build:** Rewrite the weight estimator without notes; reproduce one phone run; write a one-page explanation of what you measured.

**Verify:** A fresh shell can reproduce your command; list three unknowns and choose one to resolve next week.

**Save:** code or configuration changes, any raw observations, and `notes/day-007.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-02.md -->

# Week 02: Control flow, numeric types and reproducible experiments

Days 008–014 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md): Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 008 — Conditions and validation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** if, else, booleans, comparison, short-circuit evaluation.

**Build:** Add model-size and quantization validation to the estimator; separate validation from output.

**Verify:** Zero, negative, absurdly large and malformed input receive deliberate behavior; explain each branch.

**Save:** code or configuration changes, any raw observations, and `notes/day-008.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 009 — Loops and running statistics

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** for and while loops; counters and accumulation.

**Build:** Write a C++ latency-summary program over a fixed array; calculate count, min, max and arithmetic mean.

**Verify:** Check an empty list and a single value; compare against Python and explain why an average hides outliers.

**Save:** code or configuration changes, any raw observations, and `notes/day-009.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 010 — Floating-point behavior

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Float versus double, rounding, relative and absolute error.

**Build:** Create cancellation and summation examples; implement a numeric comparison helper.

**Verify:** Demonstrate why exact equality fails for some calculations; use a scale-aware tolerance with a near-zero case.

**Save:** code or configuration changes, any raw observations, and `notes/day-010.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 011 — Integers and size arithmetic

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Fixed-width integers, size_t, signedness and checked multiplication.

**Build:** Rework byte-size estimation with checked integer operations; retain a separate display conversion.

**Verify:** Oversized dimensions fail before wraparound; explain why casting after overflow cannot repair it.

**Save:** code or configuration changes, any raw observations, and `notes/day-011.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 012 — Strings and configuration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** std::string, parsing and error states; avoid dangling views for now.

**Build:** Parse a simple command line for model label, context and threads; keep options few and documented.

**Verify:** Unknown flags and missing values fail clearly; quoted paths are handled by the calling shell or argument list.

**Save:** code or configuration changes, any raw observations, and `notes/day-012.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 013 — Small baseline matrix

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Prefill versus decode, thread counts, fresh versus reused process.

**Build:** Extend the Python runner to three prompt lengths and two thread counts with three repetitions each.

**Verify:** Keep actual input/output counts; report individual observations plus median; do not equate fresh process with cold filesystem cache.

**Save:** code or configuration changes, any raw observations, and `notes/day-013.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 014 — Review and measurement audit

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Distinguish load, tokenization, queueing, prefill and first-token display.

**Build:** Draw the timing boundaries in prose; rewrite one function from memory and inspect every baseline log.

**Verify:** Identify one confound such as charging or prompt-template variation; correct the next experiment plan.

**Save:** code or configuration changes, any raw observations, and `notes/day-014.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-03.md -->

# Week 03: References, pointers, containers and debugging

Days 015–021 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [Clang sanitizers](https://clang.llvm.org/docs/AddressSanitizer.html): Start with AddressSanitizer; linked guides explain other diagnostics and platform restrictions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 015 — References and const

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Object identity, aliases, const references and parameter contracts.

**Build:** Implement vector sum and normalization with explicit input/output ownership; compare pass-by-value and const-reference signatures.

**Verify:** Predict which calls copy data; explain why const is an interface property and not a thread-safety guarantee.

**Save:** code or configuration changes, any raw observations, and `notes/day-015.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 016 — Pointers and lifetimes

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Addresses, dereference, null, object lifetime and storage duration.

**Build:** Write tiny safe pointer examples; deliberately inspect a dangling pointer example in an isolated failing test.

**Verify:** Explain precisely when the pointee dies; do not run known undefined behavior in normal application code.

**Save:** code or configuration changes, any raw observations, and `notes/day-016.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 017 — std::vector

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Size, capacity, contiguous storage and iterator invalidation.

**Build:** Build a vector-based dot product with dimension validation; inspect capacity changes during append.

**Verify:** Test mismatched sizes and zero length; explain why reallocation can invalidate pointers and references.

**Save:** code or configuration changes, any raw observations, and `notes/day-017.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 018 — std::array and spans

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Fixed extent versus dynamic containers; non-owning std::span.

**Build:** Implement dot-product and min/max functions taking spans so array and vector callers share code.

**Verify:** Both owners can call the same function; document that a span does not extend the underlying lifetime.

**Save:** code or configuration changes, any raw observations, and `notes/day-018.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 019 — Debugger workflow

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Breakpoints, step into/over, stack frames and variable inspection.

**Build:** Introduce an indexing bug in a small program and find it with GDB or LLDB before reading the answer.

**Verify:** Save a short debugging transcript showing the first wrong state, not merely the final crash.

**Save:** code or configuration changes, any raw observations, and `notes/day-019.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 020 — Sanitizer workflow

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** AddressSanitizer and UndefinedBehaviorSanitizer; debug symbols.

**Build:** Build isolated out-of-bounds and use-after-free tests; repair them and add regression cases.

**Verify:** The tools diagnose the intentional failures where supported; repaired normal tests run clean; keep unsafe demos separate.

**Save:** code or configuration changes, any raw observations, and `notes/day-020.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 021 — Closed-book container challenge

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Pointers, spans, lifetime, resizing and const.

**Build:** Implement cosine similarity from an empty file with explicit zero-vector behavior; review all borrowed views.

**Verify:** Match a Python oracle and explain every container choice; name the lifetime assumptions in the API.

**Save:** code or configuration changes, any raw observations, and `notes/day-021.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-04.md -->

# Week 04: Classes, RAII, moves and a minimal tensor

Days 022–028 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines): Reference for resource management, ownership and concurrency. Read a handful of applicable rules after your own design.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 022 — Structs and invariants

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Members, aggregate initialization, encapsulation and class invariants.

**Build:** Create a Shape type with checked element count and a documented policy on zero dimensions.

**Verify:** Invalid shapes cannot silently allocate a wrapped size; unit tests cover rank and zero policy.

**Save:** code or configuration changes, any raw observations, and `notes/day-022.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 023 — Constructors and RAII

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Initialization lists, destructors, resource acquisition and cleanup.

**Build:** Create an owning FP32 Tensor using vector storage and Shape; provide checked indexing.

**Verify:** Construction establishes a valid state; throwing validation does not leak resources.

**Save:** code or configuration changes, any raw observations, and `notes/day-023.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 024 — Copy and move semantics

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Value semantics, references, move operations and moved-from states.

**Build:** Instrument a small demonstration type; compare copying Tensor with moving it; prefer the rule of zero where possible.

**Verify:** Predict the observed operations before running; explain why std::move does not itself move bytes.

**Save:** code or configuration changes, any raw observations, and `notes/day-024.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 025 — Smart pointers

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** unique_ptr, shared_ptr, ownership graphs and cycles.

**Build:** Wrap a small opaque resource in unique ownership; design session and model ownership on paper.

**Verify:** No shared_ptr is added without a shared-ownership reason; destruction order is testable.

**Save:** code or configuration changes, any raw observations, and `notes/day-025.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 026 — Views and reshaping

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Owning tensors versus borrowed views; contiguity and shape compatibility.

**Build:** Implement TensorView over a span and reshape for contiguous tensors; document invalidation.

**Verify:** Reject mismatched element counts; test views while the owner is stable and demonstrate the invalidation hazard separately.

**Save:** code or configuration changes, any raw observations, and `notes/day-026.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 027 — Enums and errors

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** enum class, error categories, exceptions and alternative return values.

**Build:** Define shape, dtype and load errors; make CLI boundaries convert failures into readable diagnostics.

**Verify:** A malformed operation reports the right category; no catch-all silently converts a failure into a valid tensor.

**Save:** code or configuration changes, any raw observations, and `notes/day-027.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 028 — Tensor design review

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Invariants, rule of zero, copies, borrows and failure paths.

**Build:** Write a short API document and simplify your implementation; reconstruct TensorView without reference.

**Verify:** Explain who owns each byte and when each view becomes invalid; all tensor tests pass.

**Save:** code or configuration changes, any raw observations, and `notes/day-028.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-05.md -->

# Week 05: CMake, testing, standard algorithms and checkpoint I/O

Days 029–035 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html): Follow the tutorial matching your installed stable CMake version; latest documentation can describe newer releases.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [Clang sanitizers](https://clang.llvm.org/docs/AddressSanitizer.html): Start with AddressSanitizer; linked guides explain other diagnostics and platform restrictions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 029 — CMake targets

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Source lists, libraries, executables, include paths and target-scoped settings.

**Build:** Create a core library, CLI and test target; set your own code to C++20 without changing upstream requirements.

**Verify:** A clean out-of-tree debug build and release build both work; document exact commands.

**Save:** code or configuration changes, any raw observations, and `notes/day-029.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 030 — Testing fundamentals

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Fixtures, assertions, boundary cases and independent expected results.

**Build:** Use a lightweight test framework or CTest executable; test shapes, dot products, reshape and parsing.

**Verify:** At least one intentionally broken implementation fails a meaningful test; tests are not copies of the implementation.

**Save:** code or configuration changes, any raw observations, and `notes/day-030.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 031 — Algorithms and lambdas

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Iterators, range algorithms, captures and comparator contracts.

**Build:** Replace appropriate manual loops with standard algorithms; implement a deterministic top-k reference.

**Verify:** Tied values have documented behavior; explain any reference captured by a stored lambda.

**Save:** code or configuration changes, any raw observations, and `notes/day-031.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 032 — Templates in moderation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Function templates, type deduction and compile-time constraints.

**Build:** Generalize one small utility for float and double; compare with an ordinary overload.

**Verify:** Both instantiations are tested; remove abstraction if it obscures ownership or error behavior.

**Save:** code or configuration changes, any raw observations, and `notes/day-032.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 033 — Binary checkpoint specification

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Byte order, magic, version, dtype, dimensions, offsets and lengths.

**Build:** Write a format specification before code; export two tensors from Python with explicit endian encoding.

**Verify:** A hand-inspected byte layout matches the spec; never serialize a compiler-dependent C++ struct directly.

**Save:** code or configuration changes, any raw observations, and `notes/day-033.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 034 — Defensive checkpoint reader

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Stream I/O, file size, checked offsets and validation before allocation.

**Build:** Read the tiny format in C++; reject invalid magic, unsupported version, truncation and inconsistent tensor sizes.

**Verify:** Generate separate corrupted fixtures; each fails before an unsafe read or unreasonable allocation.

**Save:** code or configuration changes, any raw observations, and `notes/day-034.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 035 — Clean-build checkpoint drill

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Reproducible builds and cross-language agreement.

**Build:** Build from a clean directory and load Python-exported tensors; compare every value and shape.

**Verify:** A peer could reproduce using only your README; capture compiler and dependency versions.

**Save:** code or configuration changes, any raw observations, and `notes/day-035.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-06.md -->

# Week 06: Concurrency, Python bindings and systems foundations

Days 036–042 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines): Reference for resource management, ownership and concurrency. Read a handful of applicable rules after your own design.
- [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/): Focus on functions, NumPy buffers, lifetime and GIL rules; skip advanced binding features initially.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 036 — Threads and races

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Thread lifetime, join, shared state and data races.

**Build:** Write a two-thread counter example and a corrected version; keep the broken version in a separate test area.

**Verify:** Explain why apparent correct output does not prove race freedom; use ThreadSanitizer on a supported host build.

**Save:** code or configuration changes, any raw observations, and `notes/day-036.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 037 — Mutexes and condition variables

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Lock ownership, predicate waits, spurious wakeups and shutdown.

**Build:** Implement a bounded producer-consumer queue with a mutex and condition variables.

**Verify:** Empty waits, full waits and shutdown all terminate; test a waiting producer and consumer.

**Save:** code or configuration changes, any raw observations, and `notes/day-037.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 038 — Cancellation semantics

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Cooperative cancellation, request state and terminal events.

**Build:** Add request IDs and cancellation to a one-worker job executor; define queued versus running cancellation.

**Verify:** Each request emits exactly one terminal result; cancellation checks occur at documented safe points.

**Save:** code or configuration changes, any raw observations, and `notes/day-038.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 039 — Python-to-C++ boundary

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Bindings, array ownership, contiguous buffers and GIL concepts.

**Build:** Expose your dot product or Tensor operation using pybind11; reject unsupported dtype/layout rather than silently misreading it.

**Verify:** Noncontiguous input is rejected or explicitly copied; input owner stays alive during native access.

**Save:** code or configuration changes, any raw observations, and `notes/day-039.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 040 — Memory mapping and page faults

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Virtual addresses, resident pages, mmap, copy-on-write and file-backed storage.

**Build:** Add an optional read-only mapped checkpoint loader on your host; keep the stream loader for comparison.

**Verify:** Understand that mapped file size is not resident memory; validate offsets before producing views.

**Save:** code or configuration changes, any raw observations, and `notes/day-040.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 041 — Parallel work and false sharing

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Partitioned output, reductions, synchronization and cache lines.

**Build:** Parallelize independent rows of a matrix-vector operation with a simple fixed partition.

**Verify:** Matches single-thread output within tolerance; measure enough work to expose overhead and report a slower small case.

**Save:** code or configuration changes, any raw observations, and `notes/day-041.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 042 — Foundation gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Ownership, threads, binary validation and build reproducibility.

**Build:** Review your queue under stress and draw the ownership of model, session and temporary buffers.

**Verify:** No unsupported claim of production readiness; list unresolved races, lifetime risks and portability limits explicitly.

**Save:** code or configuration changes, any raw observations, and `notes/day-042.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-07.md -->

# Week 07: Python transformer mathematics and reference operations

Days 043–049 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 043 — Tensor layout and linear algebra

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Batch, sequence, hidden size, strides and matrix orientation.

**Build:** Write explicit-shape NumPy or PyTorch examples for linear projection and batched matmul.

**Verify:** Annotate every dimension and compare one tiny multiplication against a hand calculation.

**Save:** code or configuration changes, any raw observations, and `notes/day-043.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 044 — Stable softmax

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Exponent overflow, max subtraction, axis selection and masking.

**Build:** Implement stable softmax and causal masking without calling the high-level attention API.

**Verify:** Rows normalize correctly; large logits remain finite; define behavior for an all-masked row.

**Save:** code or configuration changes, any raw observations, and `notes/day-044.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 045 — Normalization

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** RMSNorm versus LayerNorm; epsilon and reduction precision.

**Build:** Implement RMSNorm with explicit dtype handling and compare to a trusted reference.

**Verify:** Test zero, constant and high-magnitude inputs; choose a numeric tolerance justified by dtype.

**Save:** code or configuration changes, any raw observations, and `notes/day-045.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 046 — Rotary positions

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Pair rotation, positions, frequencies and architecture conventions.

**Build:** Implement RoPE for a tiny tensor; isolate configuration and layout conventions.

**Verify:** Position zero behaves as expected; pair norms are preserved within rounding error; compare exact convention to the checkpoint.

**Save:** code or configuration changes, any raw observations, and `notes/day-046.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 047 — Attention reference

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Q, K, V, causal scores, scaling and head dimensions.

**Build:** Implement ordinary multi-head causal attention with explicit tensor operations.

**Verify:** Changing a future token cannot affect earlier outputs; a small example matches a manual or trusted computation.

**Save:** code or configuration changes, any raw observations, and `notes/day-047.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 048 — Feed-forward and residuals

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** SwiGLU, activation, gated projections and residual stream.

**Build:** Implement one compatible decoder block from your chosen architecture, including both residual paths.

**Verify:** Validate intermediate shapes and outputs; do not substitute architecture details without documenting them.

**Save:** code or configuration changes, any raw observations, and `notes/day-048.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 049 — Math reconstruction

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Forward-pass dependencies and numeric stability.

**Build:** Rebuild the block from a blank file, then compare it with the previous version.

**Verify:** Explain every matmul and reduction; save deterministic reference tensors for later C++ tests.

**Save:** code or configuration changes, any raw observations, and `notes/day-049.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-08.md -->

# Week 08: Checkpoint parity, tokenization and cached generation

Days 050–056 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.
- [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers/index): Read the tokenizer pipeline and use official model tokenizer assets; a hand-written educational tokenizer is a separate exercise.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 050 — Architecture contract

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Config fields, tied embeddings, attention variants and checkpoint keys.

**Build:** Select one small supported checkpoint; map each tensor name and shape to your reference architecture.

**Verify:** Stop and resolve any mismatch rather than coercing shapes; record model revision and config.

**Save:** code or configuration changes, any raw observations, and `notes/day-050.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 051 — Checkpoint loading

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** State dictionaries, dtype conversion and inference mode.

**Build:** Load weights into the Python reference; compare the first block against the official implementation.

**Verify:** Use a fixed token-ID sequence and compare named intermediates; isolate the first layer that diverges.

**Save:** code or configuration changes, any raw observations, and `notes/day-051.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 052 — Full forward parity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Final norm, output head and vocabulary logits.

**Build:** Run the full model for a short token sequence and compare logits.

**Verify:** Report max absolute error and selected logits; explain why argmax alone is insufficient validation.

**Save:** code or configuration changes, any raw observations, and `notes/day-052.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 053 — Tokenizer and chat template

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** BPE or model-specific tokenization, special tokens, BOS/EOS and templates.

**Build:** Use the official tokenizer; create fixed text-to-token fixtures including Unicode and whitespace.

**Verify:** Round trips and special-token behavior are documented; distinguish instruction template changes from model changes.

**Save:** code or configuration changes, any raw observations, and `notes/day-053.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 054 — KV cache implementation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Prefill, per-layer K/V storage, cache positions and context bounds.

**Build:** Add cache updates and one-token decode to your Python reference.

**Verify:** Cached and uncached logits agree for the same prefix within tolerance; reset actually clears session state.

**Save:** code or configuration changes, any raw observations, and `notes/day-054.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 055 — Sampling and sequence limits

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Greedy, temperature, seeded randomness, stop rules and logits processing.

**Build:** Implement greedy and basic temperature sampling; bound total context and generated tokens.

**Verify:** EOS and limit termination work; fixed-seed tests are reproducible within the controlled implementation.

**Save:** code or configuration changes, any raw observations, and `notes/day-055.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 056 — Reference freeze

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Model identity, fixtures and deterministic debug workloads.

**Build:** Freeze a small golden fixture set for operations, one block and full cached decode.

**Verify:** Write the exact numeric comparison contract; keep large weights out of your source repository.

**Save:** code or configuration changes, any raw observations, and `notes/day-056.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-09.md -->

# Week 09: C++ transformer kernels and checkpoint integration

Days 057–063 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 057 — Matrix-vector kernel

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Row-major layout, indexing and accumulation precision.

**Build:** Implement FP32 matvec against your TensorView; export random and adversarial fixtures from Python.

**Verify:** Test non-square shapes and mismatched dimensions; compare absolute and relative error.

**Save:** code or configuration changes, any raw observations, and `notes/day-057.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 058 — RMSNorm in C++

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Reduction, epsilon and output buffering.

**Build:** Port the Python RMSNorm with preallocated output and no hidden shape assumptions.

**Verify:** Match fixtures including zeros and extremes; document in-place aliasing support or reject aliasing.

**Save:** code or configuration changes, any raw observations, and `notes/day-058.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 059 — RoPE in C++

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Position indexing, frequency storage and rotation convention.

**Build:** Port the chosen model's RoPE; precompute only what the configuration requires.

**Verify:** Match Python across multiple positions; include a test that catches the wrong dimension pairing.

**Save:** code or configuration changes, any raw observations, and `notes/day-059.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 060 — Softmax and attention

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Stable exponentiation, causal bounds and score buffers.

**Build:** Implement scalar attention for one head then multiple heads; use explicit strides.

**Verify:** Match small reference tensors; verify future-token masking and finite outputs.

**Save:** code or configuration changes, any raw observations, and `notes/day-060.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 061 — Feed-forward and decoder block

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Intermediate buffers, activation and residual update order.

**Build:** Assemble one decoder block using your tested kernels.

**Verify:** Compare every stage with Python rather than debugging only final output; report first divergence.

**Save:** code or configuration changes, any raw observations, and `notes/day-061.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 062 — Checkpoint-to-native bridge

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Exported tensor names, config and endian-safe loading.

**Build:** Extend your educational format for the selected tiny checkpoint and load one complete block.

**Verify:** All shapes and lengths validated; no silent transpose; reject an unsupported architecture field.

**Save:** code or configuration changes, any raw observations, and `notes/day-062.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 063 — Block parity gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Numerical debugging and code review.

**Build:** Run all block fixtures, fix the earliest divergence and explain any tolerance change.

**Verify:** Do not move to full generation until block parity is established; list any model-specific assumptions.

**Save:** code or configuration changes, any raw observations, and `notes/day-063.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-10.md -->

# Week 10: A complete educational C++ inference engine

Days 064–070 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.
- [llama2.c](https://github.com/karpathy/llama2.c): Read only after implementing your first forward pass. This is educational C, not a production C++ architecture template.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 064 — Whole-model execution

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Layer iteration, embedding lookup, final norm and output projection.

**Build:** Assemble the complete forward pass for token IDs using your validated blocks.

**Verify:** Compare final logits for a short fixed sequence; invalid token IDs fail cleanly.

**Save:** code or configuration changes, any raw observations, and `notes/day-064.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 065 — Native KV cache

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Layout, current position, capacity and session ownership.

**Build:** Implement fixed-capacity cache allocation and append; separate immutable model weights from mutable session state.

**Verify:** Two sessions do not share cache contents; boundary and reset cases match Python.

**Save:** code or configuration changes, any raw observations, and `notes/day-065.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 066 — Prefill and decode paths

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Full prefix versus incremental step and reusable workspace.

**Build:** Implement prefill and one-token decode functions with clearly timed boundaries.

**Verify:** Cache and uncached paths agree; avoid per-token workspace growth.

**Save:** code or configuration changes, any raw observations, and `notes/day-066.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 067 — Streaming and cancellation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Token events, completion events and cooperative stop checks.

**Build:** Expose a token callback and cancellation flag in the CLI; define callback thread and lifetime.

**Verify:** Exactly one terminal status follows completion or cancellation; callbacks cannot outlive their session.

**Save:** code or configuration changes, any raw observations, and `notes/day-067.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 068 — Tokenizer integration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** External token IDs first, compatible tokenizer second.

**Build:** Keep a token-ID debug mode and integrate a trusted tokenizer interface; preserve template fixtures.

**Verify:** End-to-end text prompt produces the same input IDs as the reference; Unicode output is not split into invalid UI fragments.

**Save:** code or configuration changes, any raw observations, and `notes/day-068.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 069 — Reference implementation reading

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Small C inference code versus your C++ structure.

**Build:** Read selected llama2.c forward and generation code after writing your own; note five design differences.

**Verify:** Explain correctness or ownership implications; do not paste an implementation you cannot reconstruct.

**Save:** code or configuration changes, any raw observations, and `notes/day-069.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 070 — Engine milestone

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Parity, bounds, errors and reproducibility.

**Build:** Run a clean end-to-end generation demo and write an architecture note with model/session/workspace ownership.

**Verify:** The tiny model works on host; report supported architecture and limitations rather than claiming general GGUF compatibility.

**Save:** code or configuration changes, any raw observations, and `notes/day-070.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-11.md -->

# Week 11: Profiling and the memory hierarchy

Days 071–077 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [Android Simpleperf](https://developer.android.com/ndk/guides/simpleperf): Use supported profiling modes and document permission limits; combine with your own stage timings.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 071 — Benchmark validity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Warmup, repetition, clocks, optimizer elimination and work accounting.

**Build:** Build a kernel benchmark with checksummed output and multiple input sizes.

**Verify:** Release builds do useful work; record raw observations and separate setup from the timed region.

**Save:** code or configuration changes, any raw observations, and `notes/day-071.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 072 — Profile before changing code

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Sampling versus instrumentation, symbols and call stacks.

**Build:** Profile host decode and annotate time in matvec, attention, allocation and other work.

**Verify:** Choose the largest actionable bottleneck using evidence; save a baseline profile.

**Save:** code or configuration changes, any raw observations, and `notes/day-072.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 073 — Allocation discipline

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Capacity reuse, arenas as a concept and lifetime-bound workspaces.

**Build:** Count or instrument allocations during decode; preallocate one repeated temporary buffer.

**Verify:** Correctness remains unchanged; report allocation count and latency before and after.

**Save:** code or configuration changes, any raw observations, and `notes/day-073.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 074 — Cache locality

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Contiguous traversal, strides and working-set size.

**Build:** Compare row and column traversal on matrices that fit and exceed cache.

**Verify:** Explain observed differences without attributing every result to bandwidth; repeat to check noise.

**Save:** code or configuration changes, any raw observations, and `notes/day-074.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 075 — Bandwidth and arithmetic intensity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Bytes moved, FLOPs, roofline estimates and batch-one decode.

**Build:** Estimate weight bytes read per token and achievable bandwidth; compare with measured decode.

**Verify:** State assumptions and explain why the simple bound is not a hardware guarantee.

**Save:** code or configuration changes, any raw observations, and `notes/day-075.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 076 — Prefill versus decode workload

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Matrix-matrix versus matrix-vector shapes; amortization.

**Build:** Benchmark several prompt lengths and output lengths separately.

**Verify:** Show that one tokens-per-second number cannot describe both phases; include context length in decode results.

**Save:** code or configuration changes, any raw observations, and `notes/day-076.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 077 — Optimization proposal

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Amdahl's law, confounds and experiment scope.

**Build:** Write three optimization hypotheses ranked by impact and effort; implement only the smallest justified change.

**Verify:** A report includes unchanged correctness tests and a confidence-limited interpretation of the measurements.

**Save:** code or configuration changes, any raw observations, and `notes/day-077.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-12.md -->

# Week 12: INT8 quantization and quality evaluation

Days 078–084 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [ggml source and quantization implementation](https://github.com/ggml-org/llama.cpp/tree/master/ggml): Inspect format and kernel concepts after your scalar implementation; educational INT4 is not automatically a supported GGUF quant.
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 078 — Quantization arithmetic

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Scale, zero point, signed ranges, clipping and rounding.

**Build:** Implement scalar symmetric quantize/dequantize in Python with an explicit representable range.

**Verify:** Zero, boundaries, ties and outliers have specified behavior; test round-trip error.

**Save:** code or configuration changes, any raw observations, and `notes/day-078.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 079 — Granularity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Per-tensor, per-output-channel and group scales.

**Build:** Quantize one linear layer at each granularity and report error by output channel.

**Verify:** Scale indexing is tested; compare storage including scale overhead rather than raw integer data only.

**Save:** code or configuration changes, any raw observations, and `notes/day-079.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 080 — Native integer representation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Weight storage, accumulator width and overflow bounds.

**Build:** Write an INT8 loader and reference quantized matvec; keep the activation path explicitly documented.

**Verify:** Compare against dequantized reference; explain whether your first kernel uses integer or floating-point multiplication.

**Save:** code or configuration changes, any raw observations, and `notes/day-080.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 081 — Calibration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Representative input distribution and activation ranges.

**Build:** Collect layer statistics on a small public corpus; compare calibration sets from two domains.

**Verify:** No evaluation examples leak into tuning; report saturation and outlier effects.

**Save:** code or configuration changes, any raw observations, and `notes/day-081.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 082 — End-to-end quantized model

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Mixed precision, sensitive layers and integration.

**Build:** Replace selected projections with quantized versions; keep a float reference switch.

**Verify:** Layer and model numeric comparisons pass the chosen quality contract; quantify memory and latency.

**Save:** code or configuration changes, any raw observations, and `notes/day-082.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 083 — Task evaluation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Perplexity as one signal; application-level quality and failure cases.

**Build:** Create a fixed text evaluation subset and report float versus INT8 outcomes.

**Verify:** Keep tokenizer, prompts and decoding fixed; manually inspect examples where behavior changes.

**Save:** code or configuration changes, any raw observations, and `notes/day-083.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 084 — Quantization report

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Compression, performance and quality tradeoffs.

**Build:** Write a result table including scale overhead, artifact size, memory and task scores.

**Verify:** If INT8 is slower, explain likely unpacking or kernel overhead and retain the honest result.

**Save:** code or configuration changes, any raw observations, and `notes/day-084.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-13.md -->

# Week 13: INT4 packing, group quantization and fused dequantization

Days 085–091 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [ggml source and quantization implementation](https://github.com/ggml-org/llama.cpp/tree/master/ggml): Inspect format and kernel concepts after your scalar implementation; educational INT4 is not automatically a supported GGUF quant.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 085 — Bit operations

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Masks, shifts, sign extension and integer promotions.

**Build:** Implement encode/decode for signed 4-bit values; document your nibble ordering.

**Verify:** Exhaustively round-trip all representable values and both nibble positions; do not rely on ambiguous signed shifts.

**Save:** code or configuration changes, any raw observations, and `notes/day-085.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 086 — Packing arrays

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Odd lengths, padding and byte counts.

**Build:** Pack arbitrary-length arrays and record the original element count.

**Verify:** Lengths 0, 1, 2, 3 and large odd arrays pass; unused bits cannot produce a phantom value.

**Save:** code or configuration changes, any raw observations, and `notes/day-086.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 087 — Group quantization

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Group boundaries, scale storage and tail groups.

**Build:** Implement grouped INT4 export in Python and decode in C++.

**Verify:** Check dimensions not divisible by group size; include scale metadata in the file size.

**Save:** code or configuration changes, any raw observations, and `notes/day-087.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 088 — Reference INT4 matvec

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Unpack, rescale and accumulate without changing the model contract.

**Build:** Implement an intentionally simple correctness-first kernel.

**Verify:** Matches Python dequantized computation; document where quantization error enters.

**Save:** code or configuration changes, any raw observations, and `notes/day-088.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 089 — Fused dequantization

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Temporary buffers, repeated unpacking and data reuse.

**Build:** Compare pre-dequantized, per-call dequantized and inner-loop unpacking approaches.

**Verify:** Measure separately on small and large matrices; include temporary memory and setup time.

**Save:** code or configuration changes, any raw observations, and `notes/day-089.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 090 — Model experiment

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Sensitivity, group size and mixed precision choices.

**Build:** Run a small sweep of two group sizes and selected quantized layers.

**Verify:** Use the frozen task set and identical generation settings; do not optimize only perplexity.

**Save:** code or configuration changes, any raw observations, and `notes/day-090.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 091 — Quantization gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Compatibility versus educational formats.

**Build:** Write a precise format description and compare concepts with runtime-supported quants.

**Verify:** State that your INT4 format is educational and not automatically GGUF, QNN or hardware-native INT4.

**Save:** code or configuration changes, any raw observations, and `notes/day-091.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-14.md -->

# Week 14: ARM NEON, threading and measured optimization

Days 092–098 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Arm Neon intrinsics reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html): Look up exact intrinsics and required architecture features; preserve a scalar reference and correct tail handling.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 092 — Vectorization fundamentals

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** SIMD lanes, loads, reductions, alignment and tails.

**Build:** Inspect compiler vectorization for your scalar dot product on the target.

**Verify:** Save compiler diagnostics or assembly; identify vector and scalar remainder paths.

**Save:** code or configuration changes, any raw observations, and `notes/day-092.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 093 — First NEON kernel

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** ARM intrinsics for vector multiply-add and reduction.

**Build:** Implement one FP32 dot-product kernel with a scalar fallback.

**Verify:** Test odd lengths, tiny lengths and unaligned starting offsets where allowed; compare numerical error.

**Save:** code or configuration changes, any raw observations, and `notes/day-093.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 094 — Quantized kernel experiment

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Packed data, unpack cost and accumulation.

**Build:** Optimize one INT8 or INT4 inner loop only after the reference passes.

**Verify:** Compare output and throughput across representative shapes; keep unsupported instruction paths disabled.

**Save:** code or configuration changes, any raw observations, and `notes/day-094.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 095 — Thread count and partitioning

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Work size, scheduling overhead and heterogeneous cores.

**Build:** Sweep a few thread counts with fixed workloads and supported affinity experiments if available.

**Verify:** Do not assume all cores maximize performance; report sustained results and permitted controls.

**Save:** code or configuration changes, any raw observations, and `notes/day-095.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 096 — False sharing and buffers

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Cache-line sharing, independent output and synchronization.

**Build:** Construct a controlled false-sharing example, then revise data placement.

**Verify:** Explain why the demonstration may vary with hardware; never treat padding as a universal optimization.

**Save:** code or configuration changes, any raw observations, and `notes/day-096.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 097 — Fusion or reuse experiment

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Removing a temporary or repeated computation.

**Build:** Try one small fusion or workspace reuse based on profiling; retain reference operation tests.

**Verify:** Measure total decode impact, not just an isolated hot-loop speedup.

**Save:** code or configuration changes, any raw observations, and `notes/day-097.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 098 — Optimization milestone

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Performance claims and reproducible evidence.

**Build:** Publish a local report with one successful optimization and one failed hypothesis.

**Verify:** Every claimed improvement includes workload, build, repetitions and correctness evidence.

**Save:** code or configuration changes, any raw observations, and `notes/day-098.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-15.md -->

# Week 15: Android NDK deployment and native diagnostics

Days 099–105 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [Android Simpleperf](https://developer.android.com/ndk/guides/simpleperf): Use supported profiling modes and document permission limits; combine with your own stage timings.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 099 — Cross-compilation fundamentals

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Target triple, ABI, API level, libc and host tools.

**Build:** Create an Android CMake configuration for your educational engine; retain host tests.

**Verify:** The same source builds for host and arm64-v8a with separate build directories.

**Save:** code or configuration changes, any raw observations, and `notes/day-099.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 100 — Deploy the tiny engine

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Executable permissions, library paths and file locations.

**Build:** Push the native CLI and tiny checkpoint through ADB and run fixed token-ID input.

**Verify:** On-phone logits match the host within justified tolerance; distinguish ISA-related numeric differences from bugs.

**Save:** code or configuration changes, any raw observations, and `notes/day-100.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 101 — Symbols and crashes

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Debug symbols, backtraces and address-to-line mapping.

**Build:** Create an isolated debug crash and symbolize it; repair the bug.

**Verify:** The trace identifies your source function; do not distribute the intentional crash in the app.

**Save:** code or configuration changes, any raw observations, and `notes/day-101.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 102 — Android logs and errors

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** logcat, native error propagation and app-visible diagnostics.

**Build:** Add structured stage logs without logging document contents by default.

**Verify:** A missing checkpoint and unsupported format generate useful diagnostics and a nonzero CLI status.

**Save:** code or configuration changes, any raw observations, and `notes/day-102.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 103 — Native profiling

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Simpleperf permissions, sample collection and symbol resolution.

**Build:** Profile a debuggable or otherwise supported native target; fall back to scoped timing if restricted.

**Verify:** State which counters and traces are actually available; never fabricate unavailable hardware measurements.

**Save:** code or configuration changes, any raw observations, and `notes/day-103.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 104 — Memory accounting

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** RSS/PSS, file mappings, shared pages and allocator behavior.

**Build:** Collect available process memory information while loading, prefilling and decoding.

**Verify:** Explain why mapped bytes, RSS and PSS differ; preserve raw tool output alongside parsed values.

**Save:** code or configuration changes, any raw observations, and `notes/day-104.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 105 — Device parity gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Host-target reproducibility and compatibility.

**Build:** Write exact build, deploy and run instructions for the pinned versions.

**Verify:** Recreate the run from a clean shell; document ABI, Android version, model revision and backend.

**Save:** code or configuration changes, any raw observations, and `notes/day-105.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-16.md -->

# Week 16: Kotlin, JNI and a minimal offline application

Days 106–112 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [Android JNI tips](https://developer.android.com/training/articles/perf-jni): Read ownership, references, threads and exception guidance before integrating native generation.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 106 — Kotlin essentials

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** val/var, functions, classes, nullability and basic collections.

**Build:** Build a tiny Android screen displaying selected model metadata; avoid complex UI architecture.

**Verify:** Understand every Kotlin statement used; the app installs and runs without inference dependencies.

**Save:** code or configuration changes, any raw observations, and `notes/day-106.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 107 — JNI boundary design

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Opaque handles, ownership, native exceptions and API contracts.

**Build:** Specify create, load, generate, cancel, reset and destroy operations before implementing them.

**Verify:** Document valid call sequences and invalid handle behavior; no C++ exception crosses JNI unchecked.

**Save:** code or configuration changes, any raw observations, and `notes/day-107.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 108 — Native library integration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Shared-library packaging, ABI selection and library loading.

**Build:** Load a native library and call a small metadata or dot-product function from Kotlin.

**Verify:** Verify the packaged ABI matches the phone; failure to load is visible and diagnosable.

**Save:** code or configuration changes, any raw observations, and `notes/day-108.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 109 — Background execution

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** UI thread, worker ownership and nonblocking callbacks.

**Build:** Run a simulated token stream on a background worker and display updates.

**Verify:** UI remains responsive; callback ordering and worker lifetime are documented.

**Save:** code or configuration changes, any raw observations, and `notes/day-109.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 110 — Real streaming inference

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** JNI string/array conversion, token bytes and buffering.

**Build:** Connect the runtime and stream generated text, preserving valid UTF-8 assembly.

**Verify:** A non-ASCII prompt and output do not corrupt text; first-token display timing is measured at the UI boundary.

**Save:** code or configuration changes, any raw observations, and `notes/day-110.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 111 — Stop and cleanup

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Cooperative stop, session state and resource release.

**Build:** Implement stop, reset and model release with explicit state transitions.

**Verify:** Rapid start/stop cannot use a freed handle; exactly one terminal status reaches the UI.

**Save:** code or configuration changes, any raw observations, and `notes/day-111.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 112 — Application milestone

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Minimal surface and observable behavior.

**Build:** Demonstrate prompt, streaming output, stop and metadata in airplane mode.

**Verify:** No network service is needed for inference after assets are installed; keep logs of a complete request lifecycle.

**Save:** code or configuration changes, any raw observations, and `notes/day-112.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-17.md -->

# Week 17: Lifecycle, sustained performance and resource policies

Days 113–119 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [Android Simpleperf](https://developer.android.com/ndk/guides/simpleperf): Use supported profiling modes and document permission limits; combine with your own stage timings.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 113 — Activity recreation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** UI state versus native engine ownership.

**Build:** Test rotation or recreation during generation; decide where the engine lifetime belongs.

**Verify:** No duplicate generation or orphaned callback; explain behavior instead of suppressing lifecycle events blindly.

**Save:** code or configuration changes, any raw observations, and `notes/day-113.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 114 — Background and process loss

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Background limits, process death and recoverable state.

**Build:** Define whether generation pauses, cancels or continues only under permitted app behavior; persist lightweight metadata.

**Verify:** Relaunch is coherent after process termination; do not promise survival without the required Android execution model.

**Save:** code or configuration changes, any raw observations, and `notes/day-114.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 115 — Model switching

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Unloading order, outstanding requests and memory peaks.

**Build:** Implement sequential model switching or simulate it if only one asset is available.

**Verify:** Reject or queue new requests during switch; no callback references unloaded state.

**Save:** code or configuration changes, any raw observations, and `notes/day-115.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 116 — Sustained workload design

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Thermal state, charging, ambient conditions and run order.

**Build:** Run a 10-15 minute sequence with repeated fixed requests; log throughput and thermal status when available.

**Verify:** Plot performance over time; report test conditions and any throttling indicators without inferring unsupported sensor values.

**Save:** code or configuration changes, any raw observations, and `notes/day-116.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 117 — Energy measurement limits

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Power versus energy, battery gauge noise and external meters.

**Build:** Define an energy experiment; use coarse battery deltas only as exploratory evidence if no reliable meter exists.

**Verify:** Do not claim precise joules per token from a brief percentage change; state measurement uncertainty.

**Save:** code or configuration changes, any raw observations, and `notes/day-117.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 118 — Resource-budget policy

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Queue size, context cap, token budget and memory-pressure response.

**Build:** Implement conservative limits and a clear out-of-budget message.

**Verify:** Over-limit requests fail before excessive allocation; document how limits were chosen from measurements.

**Save:** code or configuration changes, any raw observations, and `notes/day-118.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 119 — Reliability gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Repeated cancellation, load/unload and state transitions.

**Build:** Run a structured lifecycle soak test with failure injection and raw logs.

**Verify:** No uncontrolled memory growth in available measurements; list remaining platform restrictions.

**Save:** code or configuration changes, any raw observations, and `notes/day-119.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-18.md -->

# Week 18: ExecuTorch export, graph lowering and delegation

Days 120–126 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [ExecuTorch LLM documentation](https://docs.pytorch.org/executorch/stable/llm/working-with-llms.html): Start with one supported example. Exporter, runtime and backend versions must agree.
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 120 — Export concepts

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Eager execution, captured graphs, shape constraints and graph breaks.

**Build:** Export a tiny linear model with torch.export and inspect the graph.

**Verify:** Explain parameters, inputs, outputs and shape guards; pin compatible package versions.

**Save:** code or configuration changes, any raw observations, and `notes/day-120.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 121 — Small-model deployment

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Portable runtime artifact and C++ runner.

**Build:** Follow a minimal ExecuTorch tutorial before an LLM example; deploy a tiny supported model.

**Verify:** Compare native outputs against Python; capture serialization and runtime versions.

**Save:** code or configuration changes, any raw observations, and `notes/day-121.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 122 — Operator coverage

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Supported ops, decomposition and backend requirements.

**Build:** Inspect an encoder or small transformer subgraph and list required operations.

**Verify:** Identify unsupported ops before lengthy conversion; distinguish export failure from execution failure.

**Save:** code or configuration changes, any raw observations, and `notes/day-122.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 123 — Quantized export

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Calibration, operator-level quantization and numerical checking.

**Build:** Quantize and export a small supported model following the selected backend recipe.

**Verify:** Record scale/dtype decisions and output drift; avoid assuming your educational INT4 is accepted.

**Save:** code or configuration changes, any raw observations, and `notes/day-123.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 124 — Delegation and fallback

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Partitioned graphs, data transfer and backend logs.

**Build:** Run a documented delegate example and inspect what was delegated.

**Verify:** Save evidence of actual backend execution; a successful API call alone is not proof of NPU use.

**Save:** code or configuration changes, any raw observations, and `notes/day-124.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 125 — Documented LLM attempt

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Known-supported model and end-to-end recipe.

**Build:** Attempt one pinned official LLM workflow with a predeclared time budget; preserve each failure stage.

**Verify:** If blocked, provide a minimal reproducible issue and retain the small-model success as the milestone.

**Save:** code or configuration changes, any raw observations, and `notes/day-125.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 126 — Export milestone

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Compatibility matrix and evidence.

**Build:** Write model, shape, dtype, target and delegated-op compatibility tables.

**Verify:** Separate verified support, documented-but-untested support and failed experiments.

**Save:** code or configuration changes, any raw observations, and `notes/day-126.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-19.md -->

# Week 19: Qualcomm NPU deployment and vendor runtime constraints

Days 127–133 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Qualcomm Genie LLM tutorial](https://github.com/qualcomm/ai-hub-apps/blob/release/tutorials/llm_on_genie/README.md): Target-aware vendor path; verify SDK, firmware and model-asset compatibility on your exact device.
- [ExecuTorch LLM documentation](https://docs.pytorch.org/executorch/stable/llm/working-with-llms.html): Start with one supported example. Exporter, runtime and backend versions must agree.
- [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md): Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 127 — Choose one vendor route

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Genie versus ExecuTorch Qualcomm backend; SDK and target compatibility.

**Build:** Select one supported official example and record required SDK, device family and assets.

**Verify:** Confirm documentation applies to your target or state the gap; do not mix incompatible release instructions.

**Save:** code or configuration changes, any raw observations, and `notes/day-127.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 128 — Prepare reproducible assets

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Model revision, compiler options and target-specific artifacts.

**Build:** Obtain or compile supported model assets; hash every configuration and binary used.

**Verify:** A manifest links each artifact to its source and target; respect model terms and keep credentials out of records.

**Save:** code or configuration changes, any raw observations, and `notes/day-128.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 129 — First target execution

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Runtime libraries, initialization and diagnostic logs.

**Build:** Run the smallest supported example; capture complete initialization and execution logs.

**Verify:** Classify errors as asset, ABI, library, operator or device compatibility failures.

**Save:** code or configuration changes, any raw observations, and `notes/day-129.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 130 — Numerical validation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Reference outputs and preprocessing equivalence.

**Build:** Compare vendor-run output with the appropriate reference using fixed token IDs or tensors.

**Verify:** Tokenizer and model configuration agree; task-level behavior meets a stated threshold.

**Save:** code or configuration changes, any raw observations, and `notes/day-130.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 131 — Prefill and decode configuration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Supported sequence shapes, cache layout and preparation cost.

**Build:** Vary only documented options for one model; record first-use and repeated-run behavior.

**Verify:** Do not infer unrestricted dynamic shapes from one working sequence; list actual supported configurations.

**Save:** code or configuration changes, any raw observations, and `notes/day-131.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 132 — Partitioning and overhead

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Host work, accelerator work, transfers and synchronization.

**Build:** Inspect profiling or logs for fallback and transfer boundaries; estimate their contribution only where observable.

**Verify:** Separate measured timings from inferred bottlenecks; save raw traces.

**Save:** code or configuration changes, any raw observations, and `notes/day-132.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 133 — Vendor path gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Practical support and reproducibility.

**Build:** Write a setup guide or a high-quality failure report with a CPU fallback path.

**Verify:** The roadmap continues even if OEM firmware blocks NPU access; no root requirement is invented.

**Save:** code or configuration changes, any raw observations, and `notes/day-133.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-20.md -->

# Week 20: Backend comparison and inference infrastructure

Days 134–140 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md): Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM): Optional comparison after your main runtime and application work; check supported models and packaging.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 134 — Upstream Snapdragon backend

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** CPU, Adreno and experimental Hexagon paths in the pinned runtime.

**Build:** Attempt the upstream Snapdragon path using its exact current instructions; enumerate actual devices.

**Verify:** Record what initializes and what runs; do not assume a GPU-looking offload flag proves GPU rather than another backend.

**Save:** code or configuration changes, any raw observations, and `notes/day-134.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 135 — Comparison contract

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Same-model comparisons versus whole-system comparisons.

**Build:** Define a matrix with runtime, quant, template, context, token counts and backend.

**Verify:** Mark unavoidable differences explicitly; avoid calling a different quantized model a hardware-only comparison.

**Save:** code or configuration changes, any raw observations, and `notes/day-135.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 136 — Automated matrix

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Configuration files, repetitions, timeout handling and result schema.

**Build:** Extend the Python runner to execute the small comparison matrix and resume incomplete experiments.

**Verify:** Failed runs remain visible and do not become zero latency; preserve raw logs.

**Save:** code or configuration changes, any raw observations, and `notes/day-136.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 137 — Second application runtime

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** LiteRT-LM architecture and model packaging.

**Build:** Read its official integration path; optionally run one supported sample if time remains.

**Verify:** Document tested versus untested behavior; do not expand scope if core comparisons remain incomplete.

**Save:** code or configuration changes, any raw observations, and `notes/day-137.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 138 — Request scheduling

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Single-user latency, bounded queues and admission control.

**Build:** Add one-worker queueing to your native service or app and measure queue time separately.

**Verify:** Concurrent requests cannot corrupt session state; reject overload predictably.

**Save:** code or configuration changes, any raw observations, and `notes/day-138.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 139 — Prefix reuse and cache validity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Exact token-prefix match, model identity and template identity.

**Build:** Implement a small cache-key experiment or simulation with explicit invalidation.

**Verify:** Changing model, template or token prefix prevents invalid reuse; measure benefit only for repeated prefixes.

**Save:** code or configuration changes, any raw observations, and `notes/day-139.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 140 — Deployment decision

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Latency, quality, memory, compatibility and sustained behavior.

**Build:** Choose a backend for the capstone using a decision table; include engineering complexity.

**Verify:** The choice follows your measurements; unsupported accelerators remain an optional branch.

**Save:** code or configuration changes, any raw observations, and `notes/day-140.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-21.md -->

# Week 21: Offline document assistant: retrieval and resource ownership

Days 141–147 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [SQLite documentation](https://www.sqlite.org/docs.html): Useful local metadata and index-manifest storage; vector search can start as your own exact scan.
- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 141 — Capstone scope

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Offline requirements, public data, task definition and budgets.

**Build:** Choose a small public corpus and 20 initial questions; define target memory, latency and quality envelopes.

**Verify:** Budgets are provisional hypotheses; avoid importing sensitive work documents.

**Save:** code or configuration changes, any raw observations, and `notes/day-141.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 142 — Text ingestion

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Encoding, document identity, page/source metadata and incremental updates.

**Build:** Implement text or pre-extracted PDF text import with stable source IDs.

**Verify:** Every chunk can be traced to a source location; invalid text input gets an explicit result.

**Save:** code or configuration changes, any raw observations, and `notes/day-142.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 143 — Chunking baseline

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Boundaries, overlap, token budget and provenance.

**Build:** Implement deterministic chunking and fixtures for headings, short sections and long paragraphs.

**Verify:** No dropped or duplicated source spans beyond deliberate overlap; metadata survives splitting.

**Save:** code or configuration changes, any raw observations, and `notes/day-143.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 144 — Embedding baseline

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Normalization, similarity choice and model residency.

**Build:** Deploy a small supported embedding model or precompute embeddings on host for the first baseline.

**Verify:** Label precomputed versus fully on-device stages; compare identical text preprocessing.

**Save:** code or configuration changes, any raw observations, and `notes/day-144.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 145 — Exact retrieval

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Cosine or dot similarity, top-k, stable ties and index persistence.

**Build:** Implement exact vector search over the small corpus and a persisted index manifest.

**Verify:** Compare against a Python oracle; query and index embedding versions must match.

**Save:** code or configuration changes, any raw observations, and `notes/day-145.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 146 — Answer assembly

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Prompt budget, source labels, truncation and abstention.

**Build:** Build a retrieved-context prompt with a strict token budget and requested citations.

**Verify:** Long retrieval output cannot overflow the session; missing evidence yields an abstention path.

**Save:** code or configuration changes, any raw observations, and `notes/day-146.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 147 — RAG baseline gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Traceability and stage-level measurements.

**Build:** Run the first end-to-end questions offline and save retrieval plus answer traces.

**Verify:** Report retrieval, prompt construction, prefill and decode separately; inspect cited sources manually.

**Save:** code or configuration changes, any raw observations, and `notes/day-147.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-22.md -->

# Week 22: Capstone: memory management, quality and optional multimodality

Days 148–154 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [SQLite documentation](https://www.sqlite.org/docs.html): Useful local metadata and index-manifest storage; vector search can start as your own exact scan.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 148 — Residency policy

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Embedding and generator memory, sequential loading and peak allocations.

**Build:** Compare keeping both models resident with loading one on demand.

**Verify:** Measure transition latency and available memory; choose a policy based on the phone rather than model file sizes.

**Save:** code or configuration changes, any raw observations, and `notes/day-148.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 149 — Session budgeting

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Context occupancy, source compression and history limits.

**Build:** Implement bounded history and explicit source selection; keep provenance after truncation.

**Verify:** No hidden loss of system instructions or source labels; test a full-context request.

**Save:** code or configuration changes, any raw observations, and `notes/day-149.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 150 — Index update integrity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Versioning, crash recovery and atomic replacement.

**Build:** Build a new index separately, validate it, then activate it atomically where supported.

**Verify:** Interrupted updates leave the previous valid index usable; reject embedding-version mismatch.

**Save:** code or configuration changes, any raw observations, and `notes/day-150.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 151 — Structured extraction task

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Schema validation, evidence spans and unsupported claims.

**Build:** Add a narrow document extraction task such as dates and identifiers with citations.

**Verify:** Validate schema and source evidence; invalid output cannot silently become trusted data.

**Save:** code or configuration changes, any raw observations, and `notes/day-151.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 152 — Optional audio or vision branch

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Preprocessing, encoder cost, streaming and model scheduling.

**Build:** Choose one tiny supported speech or vision example; if core work is unfinished, use the session for debugging instead.

**Verify:** Measure preprocessing and inference separately; label this as optional and never block the text capstone.

**Save:** code or configuration changes, any raw observations, and `notes/day-152.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 153 — Failure injection

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Corrupt assets, cancellation, missing index and invalid configuration.

**Build:** Create a small failure matrix and exercise recovery paths in the app.

**Verify:** Each failure has a useful user-visible result and a diagnostic log without private content.

**Save:** code or configuration changes, any raw observations, and `notes/day-153.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 154 — Resource-policy gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Quality versus memory and latency.

**Build:** Write a policy note covering residency, budgets, queueing and model choice.

**Verify:** Explain the worst observed failure and what remains unmeasured; do not claim reliability from one happy-path demo.

**Save:** code or configuration changes, any raw observations, and `notes/day-154.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-23.md -->

# Week 23: Evaluation, regression testing and sustained application behavior

Days 155–161 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): Optional reference for reproducible language-model task evaluation; your capstone still needs source-grounded custom tests.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [SQLite documentation](https://www.sqlite.org/docs.html): Useful local metadata and index-manifest storage; vector search can start as your own exact scan.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 155 — Evaluation dataset

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Answerable, unanswerable, cross-passage and distracting evidence cases.

**Build:** Expand to approximately 50 carefully reviewed questions with gold sources and scoring rubrics.

**Verify:** Separate tuning examples from a held-out subset; retain source version and author review status.

**Save:** code or configuration changes, any raw observations, and `notes/day-155.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 156 — Retrieval evaluation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Recall at k, source relevance and failure categorization.

**Build:** Calculate retrieval metrics before generation; inspect misses by chunking and embedding cause.

**Verify:** Do not credit generation for evidence the retriever never supplied; report denominators.

**Save:** code or configuration changes, any raw observations, and `notes/day-156.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 157 — Answer and citation scoring

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Correctness, support, citation validity and appropriate abstention.

**Build:** Score answers with a human-readable rubric; use any model judge only as a secondary signal.

**Verify:** Manually audit disagreements and representative failures; distinguish correctness from fluent phrasing.

**Save:** code or configuration changes, any raw observations, and `notes/day-157.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 158 — Performance evaluation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Stage timings, percentile estimates and sustained runs.

**Build:** Run the fixed set under controlled conditions and preserve per-request records.

**Verify:** Small samples are not sold as reliable tail percentiles; include individual failures and warmup policy.

**Save:** code or configuration changes, any raw observations, and `notes/day-158.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 159 — Regression suite

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Frozen fixtures, acceptable drift and compatibility gates.

**Build:** Create release checks for tokenization, numeric parity, schema output and lifecycle behavior.

**Verify:** A deliberately changed template or bad model manifest is detected; tolerances have a documented reason.

**Save:** code or configuration changes, any raw observations, and `notes/day-159.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 160 — Ablation study

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** One variable at a time and interaction effects.

**Build:** Compare two context budgets or retrieval settings using the held-out protocol.

**Verify:** Report quality and latency together; avoid selecting a winner on the same examples used repeatedly for tuning.

**Save:** code or configuration changes, any raw observations, and `notes/day-160.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 161 — Evaluation gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Evidence-based claims and practical limits.

**Build:** Write a report with failures, sample sizes and the chosen operating point.

**Verify:** All tables can be regenerated from saved records; conclusions fit the measured scope.

**Save:** code or configuration changes, any raw observations, and `notes/day-161.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: weeks/week-24.md -->

# Week 24: Release engineering, portfolio and the next specialization

Days 162–168 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Android application fundamentals](https://developer.android.com/guide/components/fundamentals): Review lifecycle and packaging requirements for your application. This curriculum is not a Play Store publishing checklist.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 162 — Model packaging

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Hashes, manifests, compatibility, licenses and asset versioning.

**Build:** Define a manifest and validate required files, runtime compatibility and hashes before activation.

**Verify:** A tampered asset is rejected; include model origin and required terms without redistributing restricted weights.

**Save:** code or configuration changes, any raw observations, and `notes/day-162.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 163 — Interrupted installation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Temporary downloads/copies, validation and atomic activation.

**Build:** Implement interrupted asset-install recovery using small fixtures before large files.

**Verify:** The old valid model remains usable; incomplete assets are never selected as active.

**Save:** code or configuration changes, any raw observations, and `notes/day-163.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 164 — Clean build and installation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Dependency pinning, reproducible instructions and device setup.

**Build:** Build from a clean directory and install using only documented steps.

**Verify:** Record any manual step you previously forgot; avoid claiming bit-for-bit reproducibility unless measured.

**Save:** code or configuration changes, any raw observations, and `notes/day-164.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 165 — Soak and lifecycle release check

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Repeated requests, cancellations, model switching and process restart.

**Build:** Run a bounded scripted reliability session and inspect memory plus errors.

**Verify:** Every failure is triaged; unresolved issues appear in known limitations rather than being hidden.

**Save:** code or configuration changes, any raw observations, and `notes/day-165.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 166 — Portfolio documentation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Architecture, tradeoffs, measurements and readable code.

**Build:** Write three READMEs for the educational engine, benchmark suite and capstone; include one optimization case study.

**Verify:** Another engineer can find the important code and reproduce a result without chat history.

**Save:** code or configuration changes, any raw observations, and `notes/day-166.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 167 — Upstream contribution preparation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Minimal reproduction, regression tests and project conventions.

**Build:** Prepare a small local patch, issue draft or documentation correction for a real observed problem.

**Verify:** No external message is sent automatically; your contribution is specific, reproducible and respectful of project scope.

**Save:** code or configuration changes, any raw observations, and `notes/day-167.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 168 — Final reconstruction and next plan

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Independent understanding and specialization choice.

**Build:** Explain the complete request path; reimplement one critical kernel from memory; choose ARM, NPU/compiler or runtime infrastructure for the next eight weeks.

**Verify:** Score the milestone rubric honestly; produce a concrete next-month plan based on the weakest demonstrated skill.

**Save:** code or configuration changes, any raw observations, and `notes/day-168.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)


<!-- Source: guides/android-and-backends.md -->

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


<!-- Source: guides/benchmarking.md -->

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


<!-- Source: guides/capstone.md -->

# Capstone: a measured offline document assistant

## Product boundary

Import a small public or synthetic document collection. Retrieve relevant passages locally. Generate a short answer with source references using an on-device model. Show a useful failure when evidence or resources are insufficient. Work in airplane mode after assets are installed.

Begin with text or pre-extracted PDF content. Your existing OCR experience is an advantage, but adding complex page reconstruction too early would obscure the new inference learning. Full OCR, large tables and multimodal reasoning are optional later branches.

## Acceptance levels

**Level 1:** native LLM generation on the phone with manually supplied context. **Level 2:** local index and retrieval with host-precomputed embeddings. **Level 3:** embeddings and generation both on-device. **Level 4:** lifecycle reliability, measured budgets and repeatable evaluation. Clearly state the achieved level rather than presenting host-precomputed work as fully on-device computation.

## Data model

Keep Document records with stable ID, source name, content hash and import version. Keep Chunk records with document ID, source location, text and chunking version. Keep an IndexManifest with embedding model/revision, normalization policy, dimensions, metric and corpus hash. Keep RequestTrace records with retrieved chunk IDs, prompt token count, timings and outcome.

Use a local store such as SQLite for metadata if useful. For a small corpus, an exact vector scan is an understandable and sufficient baseline. Add approximate indexing only after measuring the exact baseline and its quality.

## Request path

Validate request and budgets; embed the query; retrieve candidates; choose source passages under the context budget; construct a versioned prompt; generate with bounded output; validate references or structured output; return answer and evidence. Record per-stage timings and explicit terminal status.

Retrieved text is evidence, not instructions for your application. Keep source content separate from your control prompt and test documents containing instruction-like text. The application's answer policy should preserve source provenance and allow abstention.

## Resource policies to implement

- One generation worker initially, with bounded queue capacity.
- Explicit per-request input and output token limits.
- No model switching while an active session still owns its buffers.
- Conservative embedding/generator residency based on measured peaks.
- Cancellation at known safe points with a single terminal event.
- Recovery after process recreation without assuming native state survives.
- Temporary asset/index writes followed by validation and activation.

## Evaluation set

Create approximately 50 reviewed questions. Suggested composition: 20 direct factual, 10 requiring multiple passages, 10 unanswerable, 5 with similar distracting facts and 5 long-context or formatting-sensitive cases. Adjust to the corpus; the goal is coverage, not a universal benchmark ratio.

For each item store question, answerability, expected facts, gold source locations, acceptable answer variation and scoring notes. Keep a held-out subset. Do not repeatedly tune on it and still describe it as untouched evaluation.

## Quality measures

Retrieval recall at k asks whether required evidence is present in the selected set. Answer correctness checks expected facts. Support rate asks whether claims follow from evidence. Citation validity checks that references exist and point to supporting passages. Abstention scoring distinguishes appropriately declining from failing an answerable question.

Use clear denominators: five supported claims out of six evaluated claims differs from five fully supported answers out of six questions. Report human review and any judge model separately. Preserve disagreement examples.

## Performance measures

Report ingestion/index time separately from query time. Split query embedding, retrieval, prompt assembly, load/preparation, prefill and decode. Record application TTFT and total latency, context size, output size, available memory measurements and failure count. Include a sustained workload rather than only a single cool-device request.

Targets should be selected after your baseline. Do not invent a guaranteed token rate from the chipset specification. A smaller model with better retrieval and predictable latency may be the best deployment even if a larger model fits.

## Release demonstration

From a clean launch: select a valid installed model; import or load the corpus; ask an answerable question; inspect citations; ask an unanswerable question; cancel a long request; restart the application; simulate a missing or corrupt asset. Save the exact build and evaluation metadata.

## Portfolio write-up

Explain the problem, architecture, model/backend decision, memory ownership, benchmark method, quality results and observed failures. Include one concrete before/after optimization with raw evidence. Distinguish your educational engine from the established runtime used to deliver the final app.

The capstone is strong when another developer can reproduce a limited, well-defined result. It does not need to claim universal document understanding or perfect extraction.


<!-- Source: guides/cpp-learning-map.md -->

# C++ from the beginning to inference-oriented engineering

## How this differs from a general C++ course

You will learn the language through arrays, model files, kernels and request processing. That still requires ordinary language fundamentals: skipping types, initialization, object lifetime or error handling because you already know Python produces fragile native code.

Use LearnCpp's index to locate the named topics. Do not assume Python reference semantics, integer behavior or memory safety carry over. C++ gives you both useful control and more ways to make an invalid program.

## Stages and evidence

| Stage | Required concepts | Evidence |
|---|---|---|
| Weeks 1–2 | Compilation/linking, initialization, functions, types, conditions, loops, strings, numeric limits | A robust CLI and numeric estimator |
| Week 3 | References, pointers, const, containers, spans, lifetime, debugging | Correct array operations and diagnosed memory errors |
| Week 4 | Structs/classes, constructors, destructors, invariants, RAII, copies, moves, unique ownership | Tensor and TensorView with explicit contracts |
| Week 5 | CMake, unit tests, algorithms, lambdas, basic templates, binary I/O | Validated checkpoint loader |
| Week 6 | Threads, mutexes, condition variables, cancellation, mappings, Python bindings | Bounded executor and reference parity |
| Weeks 7–10 | Interfaces, integration, numerical implementation and ownership across modules | Small native transformer |
| Weeks 11–14 | Allocation behavior, layout, vectorization, intrinsics, profiling | Evidence-backed optimization |
| Weeks 15–24 | ABI, shared libraries, JNI, lifecycle, deployment and release contracts | Reliable Android integration |

## The essential mental models

**Object versus storage:** storage can exist without a live object of the type you want. Pointer casting does not establish alignment, lifetime or valid representation. For binary checkpoints, decode fields explicitly; do not assume mapped bytes can always be reinterpreted as arbitrary C++ objects.

**Ownership versus access:** a vector may own memory while a span borrows it. A function accepting a view must document that the owner remains alive and stable. Vector reallocation can invalidate a view even when the vector object itself remains alive.

**Value versus identity:** copying a Tensor can copy its data if it has value semantics. Sharing immutable weights can instead be appropriate at the model layer. Make these decisions explicit; a smart pointer is not a substitute for an ownership design.

**Const versus concurrency:** a const method can still touch shared external state or mutable members. Thread safety follows synchronization and ownership rules, not spelling alone.

**Move versus copy:** std::move expresses permission to move through overload resolution; it does not itself transfer data. Do not move from a resource and then assume its previous contents remain accessible. Prefer standard containers and the rule of zero before writing custom special member functions.

**Exceptions versus boundaries:** RAII should clean resources during failures. At a C or JNI boundary, translate exceptions into an explicit error contract. Do not let a native exception escape through an interface that cannot handle it.

## Topics to add when motivated

Basic inheritance and virtual functions are useful for a backend interface, but you do not need a deep hierarchy for Tensor. Compare a virtual interface, a tagged variant and a plain function table on clarity before worrying about nanoseconds. Study templates only far enough to understand the generic code you use; advanced expression templates are optional.

Learn std::optional and std::variant when designing missing values or tagged results. std::expected is a later-standard option; do not introduce it into C++20 code without a deliberate compatible implementation. Read atomics and the memory model before designing a lock-free queue. A mutex-based bounded queue is sufficient for the main course.

## Recommended learning checks

- Explain which objects exist at every line of a function returning a view.
- Explain why a signed integer overflow is not a valid wraparound technique.
- Predict when a vector append invalidates an address.
- Distinguish a link error from a compile error and a loader error.
- Diagnose an out-of-bounds access using tools rather than adding random prints.
- Explain how a waiting worker exits during shutdown.
- Explain whether a Tensor copy is intended, expensive or forbidden.
- Review a binary loader as if its input were corrupted.

## Definition of production progress

After six months, a reasonable result is well-structured small native components, good debugging discipline and a credible Android capstone. Large-scale production C++ expertise still requires code review, maintenance and real failure exposure. Use the rubric to show evidence instead of awarding yourself a title after a fixed number of days.


<!-- Source: guides/lab-contracts.md -->

# Implementation lab contracts

These are specifications, not completed solutions. Daily sessions build them incrementally. Keep reference versions when optimizing.

## Lab A: Device benchmark runner

**Purpose:** turn an ad hoc phone experiment into a reproducible measurement.

Inputs: device selector, runtime path, model identifier/hash, backend, prompt fixture, generation settings, repetition count and timeout. Use subprocess argument arrays. The runner must not interpret prompt text as shell code.

Outputs: raw stdout/stderr, exact arguments, elapsed time, return code, run identity and parsed metrics when available. Separate collection from parsing so parser bugs do not destroy evidence. If a process times out, preserve the partial log and label the run incomplete.

Minimum tests: no device; multiple devices without explicit selection; missing executable; model load failure; malformed runtime output; timeout; successful output; interrupted experiment resume. Parser tests can use small saved log fixtures without a phone.

Acceptance: replay one run from its metadata and regenerate the summary from raw records. Do not impute failed runs as zero latency or quietly remove them.

## Lab B: Tensor owner and view

**Purpose:** learn shape, storage and lifetime contracts.

The owner stores contiguous FP32 values and validated dimensions. The view borrows contiguous storage with compatible shape. Start with row-major layout and a small supported rank. Define zero-sized dimension behavior explicitly.

Operations: element count, checked indexing, reshape preserving element count, dot product and matrix-vector multiplication. Validate multiplication of dimensions before allocation. A reshape changes interpretation, not the underlying count.

Minimum tests: one-dimensional, rectangular and zero-sized cases; invalid index; mismatched dot product; dimension multiplication overflow; view invalidation demonstration in an isolated negative test. Normal application tests must not intentionally execute undefined behavior.

Acceptance: describe exactly who owns storage and whether an API can return a view to a local owner. Reject unsupported striding instead of pretending every input is contiguous.

## Lab C: Checkpoint format and loader

**Purpose:** bridge Python reference data and native code without trusting binary input.

Write a short format specification with magic, version, byte order, tensor count, names, dtype, dimensions and byte lengths. Either store offsets explicitly or define an unambiguous sequential layout. Encode integer fields at fixed widths. Do not dump a host C++ struct and assume it is portable.

Validate the file's total length, every field length, tensor size arithmetic, offset ranges and duplicate names. Bound metadata sizes before allocating. Define alignment requirements for mapped access; copying to properly aligned storage is a valid first implementation.

Minimum tests: bad magic/version; truncated header; truncated data; impossible rank; enormous dimension; duplicate tensor name; unsupported dtype; invalid offset; overlapping ranges if prohibited. A Python exporter creates valid fixtures independently of the C++ parser.

Acceptance: rejected inputs fail with a category and useful detail. The loader cannot return a tensor view whose mapping has already been destroyed.

## Lab D: Small decoder-only engine

**Purpose:** understand the full inference path, not reproduce all features of a mature runtime.

Support exactly one architecture/configuration first. Define embedding, norm, attention, RoPE, MLP, residual and output-head behavior from that architecture. Use fixed token IDs for initial parity. Keep model weights immutable and session cache/workspace mutable.

Minimum tests: each operation, one block, full short prefix, cached versus uncached logits, multiple cache positions, reset, invalid token, full context, EOS, cancellation and independent sessions. Save intermediate tensors for the first divergence rather than judging only generated prose.

Acceptance: model revision and format are recorded; numerical agreement is demonstrated with a justified tolerance; native generation works on host and later Android. General model support is explicitly out of scope.

## Lab E: Quantized matvec

**Purpose:** understand representation, dequantization cost and quality loss.

Implement symmetric INT8 and grouped packed INT4 using a specified signed range, rounding rule, scales, packing order and tail policy. An all-zero group needs a safe scale rule. Keep a scalar dequantized oracle separate from the optimized path.

Minimum tests: every 4-bit code; both nibble positions; odd lengths; tail groups; zeros; clipped outliers; non-square matrices; scale indexing; large accumulations. Measure scales and metadata in storage totals.

Acceptance: errors versus original float weights are separated from errors versus the dequantized reference. Report unpacking time, workspace and overall model performance rather than claiming speedup from smaller weights alone.

## Lab F: Bounded executor

**Purpose:** learn reliable inference infrastructure with limited concurrency.

States: queued, running, cancellation requested, and one terminal result such as completed/cancelled/failed. Decide whether cancelling a queued item removes it immediately or marks it for skipping. Define shutdown as accepting no new work, waking waiters, and finishing or cancelling existing work according to one documented policy.

Minimum tests: queue full; worker waiting; producer waiting; cancel before start; cancel while running; callback throws or fails; shutdown with pending work; repeated shutdown; rejected new work. Callbacks must not run while holding a lock if they can re-enter your executor.

Acceptance: each accepted request has exactly one terminal event and no callback outlives its receiver. ThreadSanitizer is useful on supported host builds but does not prove all possible schedules are safe.

## Lab G: Native Android shell

**Purpose:** host inference without freezing or corrupting the application.

Keep a small JNI interface and document handle ownership, permitted call order, callback thread and cancellation behavior. Do not pass raw pointers as unvalidated public handles without a lifetime strategy. Keep model sessions away from the UI thread.

Minimum tests: repeated start/stop; invalid handle; load failure; Unicode output; screen recreation; backgrounding; model switching; process restart. Preserve valid text when tokens contain partial UTF-8 sequences.

Acceptance: terminal events are coherent, state is recoverable, and diagnostics are useful. Android lifecycle policy is explicit rather than relying on a lucky activity lifetime.

## Lab H: Backend deployment investigation

**Purpose:** prove what runs, where, and with what limits.

Use an official supported example before a custom model. Record model/config/SDK/runtime/device versions. Validate outputs and inspect delegation or runtime logs. A successfully loaded accelerator library is not proof that the main graph executes there.

Minimum evidence: initialization log, executed model, input shape, target backend, quantization, output check, preparation cost and steady-state measurement. If blocked, provide a minimal reproduction and CPU fallback.

Acceptance: classify support as verified, documented but untested, or failed on this configuration. A careful incompatibility report is a valid result.


<!-- Source: guides/milestones.md -->

# Milestones and honest self-assessment

Score each area 0–3: **0** not attempted; **1** runs with substantial help or unexplained behavior; **2** independently implemented and tested; **3** independently explained, reconstructed and investigated under failures. A total is less important than unresolved zeros in prerequisites.

## Week 6: Native foundations

Required evidence: a clean host build, Tensor/Shape contracts, validated binary reader, independent fixtures, sanitizer-based debugging and a bounded worker queue. Explain a borrowed view's lifetime, a move, integer overflow and worker shutdown. If you cannot do this, spend another one or two weeks repairing foundations before the native transformer.

## Week 10: Inference understanding

Required evidence: Python reference, full C++ forward pass, intermediate parity, working KV cache, bounded generation and a small checkpoint. Explain each tensor dimension, why prefill differs from decode, and how cache positions affect RoPE. An attractive chat demo without parity evidence does not pass.

## Week 14: Optimization discipline

Required evidence: valid release benchmarks, profile, INT8 and INT4 references, at least one ARM optimization attempt, and preserved correctness. Explain why one optimization failed. You can pass with modest speedups if the explanation and measurements are sound.

## Week 17: Android reliability

Required evidence: native deployment, minimal app, working cancellation, coherent lifecycle handling, diagnostics and sustained measurements. Explain the callback lifetime and model/session ownership. A crash or deadlock on repeated start/stop requires repair before the capstone.

## Week 20: Backend investigation

Required evidence: one verified deployment path, one export or accelerator investigation, output validation and a versioned compatibility matrix. A well-documented NPU incompatibility can pass if CPU deployment works and your conclusions are bounded by evidence. Do not reward unsupported hardware claims.

## Week 24: Integrated engineering

Required evidence: offline application, documented on-device stages, fixed evaluation set, resource policies, failure injection, clean-build instructions and release metadata. Explain why the chosen model/backend is appropriate and what the application cannot do reliably.

## Oral questions

1. Which allocations scale with parameters, context, batch and concurrent sessions?
2. What is stored in the KV cache, and what makes it invalid for reuse?
3. How would you distinguish a tokenizer bug from a matrix kernel bug?
4. Why might an INT4 model be slower than INT8?
5. When does a vector invalidate a TensorView?
6. How do you cancel a worker without freeing memory it still uses?
7. What proves the expensive graph actually executes on the NPU?
8. Why is process restart not a guaranteed cold-storage benchmark?
9. What do your memory measurements count and potentially overlap?
10. Which quality failures are caused by retrieval versus generation?

## Portfolio completion rubric

For each of the three projects score correctness, ownership, reproducibility, measurement, failure handling and explanation. Aim for level 2 across all areas and level 3 in selected areas. Identify the weakest demonstrated area and use it to choose the next specialization. Do not confuse a fixed calendar with a guarantee of production mastery.


<!-- Source: guides/next-eight-weeks.md -->

# Optional weeks 25–32: choose one specialization

Do not attempt every track at once. Retain the same daily structure: recall, focused reading, implementation, verification and a note. Each week use six build sessions and a seventh reconstruction/review session. These are extension outlines, not additional numbered daily assignments.

## Track A: ARM CPU kernels

| Week | Work | Evidence |
|---|---|---|
| 25 | Establish supported CPU features and inspect generated assembly | Feature matrix and one explained inner loop |
| 26 | Optimize packed quantized dot products | Scalar parity and shape sweep |
| 27 | Explore weight packing and data layout | Repacking cost versus repeated-use benefit |
| 28 | Implement cache blocking for a selected matmul workload | Working-set explanation and measured comparison |
| 29 | Investigate threading and scheduling | Small/large workload crossover |
| 30 | Try a justified fusion | End-to-end benefit and numeric contract |
| 31 | Integrate one improvement into a runtime branch | Focused regression tests |
| 32 | Prepare a contribution and report | Reproducible patch or issue draft |

## Track B: NPU export and compiler deployment

| Week | Work | Evidence |
|---|---|---|
| 25 | Capture and inspect transformer graphs | Operator and shape inventory |
| 26 | Trace decomposition and lowering | Before/after graph explanation |
| 27 | Study backend quantization requirements | Calibration and quality comparison |
| 28 | Measure partition boundaries and fallback | Verified delegated operations |
| 29 | Explore supported sequence/cache configurations | Compatibility and memory table |
| 30 | Optimize preparation and repeated use | Initialization versus steady-state results |
| 31 | Minimize a real unsupported case | Small reproducible model |
| 32 | Contribute a test, fix or documentation improvement | Specific evidence-backed contribution |

## Track C: Runtime infrastructure — recommended first for your background

| Week | Work | Evidence |
|---|---|---|
| 25 | Model/session/workspace accounting | Allocation inventory and limits |
| 26 | Admission control and bounded queues | Queue-time and overload tests |
| 27 | Exact prefix reuse and invalidation | Correctness and repeated-prefix benefit |
| 28 | Simulate block-based KV allocation | Fragmentation and capacity analysis |
| 29 | Study continuous batching on your server environment | Scheduler trace with realistic requests |
| 30 | Explore speculative decoding with a supported runtime | Acceptance rate and total latency, including drafter cost |
| 31 | Study model switching and state restoration | Failure-safe transitions and memory peaks |
| 32 | Implement one small runtime improvement | Tests and an end-to-end comparison |

The simulation in week 28 is educational; implementing a general paged attention kernel is a separate project. Server batching experiments belong on an appropriate server/runtime, not necessarily the phone. CUDA is useful for this wider inference track but does not execute on the phone's Adreno or Hexagon hardware.

## Optional multimodal project

Choose speech recognition plus generation, a small image encoder plus grounded answers, or local OCR-assisted extraction. Establish standalone stage correctness before composing models. Measure preprocessing, encoder, decoder, buffering, memory transitions and task quality. Reuse the capstone's cancellation and asset policies rather than building a second application framework.


<!-- Source: guides/numerics-and-quantization.md -->

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


<!-- Source: guides/production-and-debugging.md -->

# Production engineering and debugging playbook

## Ownership design before performance

Use an immutable Model for weights and configuration, a Session for cache and position, a Workspace for temporary buffers, and a Request for input/output limits and cancellation. This is a useful educational split, not a mandatory class hierarchy.

Models may be shared if the runtime supports that safely; mutable sessions should not be shared accidentally. Borrowed views must not outlive weights, mappings or workspace owners. Document whether callbacks execute synchronously, on a worker, or on a UI dispatcher.

## Error taxonomy

Distinguish invalid request, invalid asset, unsupported configuration, resource limit, cancelled operation, backend initialization failure and internal failure. Preserve diagnostic detail for developers while showing actionable short messages to users. Do not convert every error into an empty successful answer.

At ABI boundaries, contain C++ exceptions. Clean resources with RAII. Avoid destructors that throw. Use a deliberate result protocol where the interface requires one. A catch-all is a final containment boundary, not the normal implementation of error handling.

## Debugging ladder

1. Reproduce with the smallest input and stable versions.
2. Classify compile, link, loader, runtime, numeric or lifecycle failure.
3. Remove accelerator and concurrency variables if possible.
4. Find the first wrong state or tensor, not the last visible symptom.
5. Fix the root cause and add a regression case.
6. Restore one variable at a time and remeasure.

### Compiler error

Read the first relevant diagnostic, locate the smallest failing expression and verify types. Do not repeatedly rewrite surrounding code. A template error can originate at your call site even if the longest message comes from a library header.

### Undefined symbol or library load failure

Check target ABI, link inputs, exported symbols, runtime library paths and dependency versions. A host library cannot simply be copied into an Android ARM64 process. Explain the distinction between a successful build and a successful target load.

### Segmentation fault

Use symbols and a debugger or sanitizer on a reproducible host case. Inspect bounds, lifetime, invalidated views and alignment. Increasing memory allocation randomly is not a diagnosis.

### Incorrect tokens

Compare token IDs and special tokens first, then architecture/config, embeddings, first layer, final logits and cache position. Disable sampling and use fixed input IDs. A different chat template can change output without a numerical kernel bug.

### Slow decode

Check release build, actual backend, thread settings, allocations, context length, fallback, thermal conditions and synchronization. Profile before changing kernels. A larger model or quant that fits can still be slower because of data movement or unsupported compute.

### Freeze on stop or shutdown

Check lock order, waits without predicates, callbacks invoked under locks and workers waiting on a queue that never receives a shutdown notification. Define state transitions and terminal events before adding timeouts everywhere.

### Growing memory

Separate intentional model/cache residency from leaked ownership. Inspect repeated load/unload and request cycles, outstanding callbacks, retained JNI references and queues. Allocators may retain freed blocks, so stable high RSS is not automatically a leak. Look for unbounded growth and ownership evidence.

## Release checks

Keep host numerical tests and boundary tests fast. Add targeted Android lifecycle tests for behavior that cannot be established on host. Save model/runtime compatibility metadata and validate assets before activation. Download into a temporary area, verify, then activate atomically where the filesystem/API permits it.

Separate correctness, performance and quality regressions. Performance thresholds should account for measurement variance and device state; avoid a brittle 'must be faster every run' check. Quality evaluation should retain examples that previously failed.

## What not to overbuild

You do not need Kubernetes, distributed tracing infrastructure, microservices or a network server for a phone-only learning app. A local structured log, simple benchmark runner, small native API and explicit state machine can expose the relevant systems engineering clearly.

Production quality comes from maintained contracts and evidence, not adding infrastructure names. Your existing deployment experience is useful when deciding how little machinery the target actually needs.


<!-- Source: resources.md -->

# Reading and source catalogue

Checked core references and curated supplementary entry points. Dynamic pages can change; pin software versions and keep the matching documentation. Read by topic rather than treating every linked site as required cover-to-cover reading.

## cpp: C++ language curriculum

https://www.learncpp.com/

Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.

## guidelines: C++ Core Guidelines

https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines

Reference for resource management, ownership and concurrency. Read a handful of applicable rules after your own design.

## cmake: CMake tutorial

https://cmake.org/cmake/help/latest/guide/tutorial/index.html

Follow the tutorial matching your installed stable CMake version; latest documentation can describe newer releases.

## debug: Clang sanitizers

https://clang.llvm.org/docs/AddressSanitizer.html

Start with AddressSanitizer; linked guides explain other diagnostics and platform restrictions.

## android: Android NDK

https://developer.android.com/ndk/guides

Primary reference for native builds, ABI, native APIs and tooling.

## jni: Android JNI tips

https://developer.android.com/training/articles/perf-jni

Read ownership, references, threads and exception guidance before integrating native generation.

## llama: llama.cpp Snapdragon guide

https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md

Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.

## bench: llama.cpp repository and benchmark sources

https://github.com/ggml-org/llama.cpp

Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

## perf: Android Simpleperf

https://developer.android.com/ndk/guides/simpleperf

Use supported profiling modes and document permission limits; combine with your own stage timings.

## pybind: pybind11 documentation

https://pybind11.readthedocs.io/en/stable/

Focus on functions, NumPy buffers, lifetime and GIL rules; skip advanced binding features initially.

## torch: PyTorch documentation

https://docs.pytorch.org/docs/stable/index.html

Use tensor, inference-mode and export API references matching your installed release.

## model: SmolLM2-135M official model card

https://huggingface.co/HuggingFaceTB/SmolLM2-135M

An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.

## tokenizer: Hugging Face Tokenizers documentation

https://huggingface.co/docs/tokenizers/index

Read the tokenizer pipeline and use official model tokenizer assets; a hand-written educational tokenizer is a separate exercise.

## llama2c: llama2.c

https://github.com/karpathy/llama2.c

Read only after implementing your first forward pass. This is educational C, not a production C++ architecture template.

## quant: ggml source and quantization implementation

https://github.com/ggml-org/llama.cpp/tree/master/ggml

Inspect format and kernel concepts after your scalar implementation; educational INT4 is not automatically a supported GGUF quant.

## arm: Arm Neon intrinsics reference

https://arm-software.github.io/acle/neon_intrinsics/advsimd.html

Look up exact intrinsics and required architecture features; preserve a scalar reference and correct tail handling.

## executorch: ExecuTorch LLM documentation

https://docs.pytorch.org/executorch/stable/llm/working-with-llms.html

Start with one supported example. Exporter, runtime and backend versions must agree.

## qnn: Qualcomm Genie LLM tutorial

https://github.com/qualcomm/ai-hub-apps/blob/release/tutorials/llm_on_genie/README.md

Target-aware vendor path; verify SDK, firmware and model-asset compatibility on your exact device.

## litert: LiteRT-LM

https://github.com/google-ai-edge/LiteRT-LM

Optional comparison after your main runtime and application work; check supported models and packaging.

## rag: SQLite documentation

https://www.sqlite.org/docs.html

Useful local metadata and index-manifest storage; vector search can start as your own exact scan.

## eval: lm-evaluation-harness

https://github.com/EleutherAI/lm-evaluation-harness

Optional reference for reproducible language-model task evaluation; your capstone still needs source-grounded custom tests.

## release: Android application fundamentals

https://developer.android.com/guide/components/fundamentals

Review lifecycle and packaging requirements for your application. This curriculum is not a Play Store publishing checklist.

## Additional primary references

- Android native cross-compilation: https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md
- ExecuTorch Qualcomm backend: https://docs.pytorch.org/executorch/stable/android-qualcomm.html
- Android LLM runner: https://docs.pytorch.org/executorch/stable/llm/run-on-android.html
- Perfetto: https://perfetto.dev/docs/
- UndefinedBehaviorSanitizer: https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
- ThreadSanitizer: https://clang.llvm.org/docs/ThreadSanitizer.html
- C++ API reference: https://en.cppreference.com/w/
- Google Benchmark: https://github.com/google/benchmark
- Catch2: https://github.com/catchorg/Catch2
- Llama 3.2 1B Instruct model card: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Gemma 3 1B IT model card: https://huggingface.co/google/gemma-3-1b-it

These models are stable educational candidates, not a ranking of the newest checkpoints. Architecture, licensing and backend support should determine the final selection. The curriculum's small decoder implementation targets exactly one chosen architecture initially.

## How to read source code

Start at the CLI entry and follow one request. Identify model loading, tokenization, batch construction, graph execution, cache mutation and sampling. Read a single operator implementation with its test. Write a call-path note before reading another subsystem. For a large file, search for the symbols on your call path rather than reading every line sequentially.

For papers, write the problem, assumptions, algorithm, implementation implications and one experiment you could reproduce. Save elaborate paper study for the optional specialization stage if it blocks the core implementation.


<!-- Source: templates/daily-log.md -->

# Day NNN — topic

Date:
Actual minutes:
Core task completed / partial / blocked:

## Recall before looking at code

What I can explain from yesterday:
One function or idea I reconstructed:

## Contract

Inputs, shapes and dtypes:
Owners and borrowed lifetimes:
Outputs and failure behavior:

## Work

Code/configuration changed:
Exact commands or run IDs:
Reading used:

## Verification

Independent oracle or expected result:
Boundary cases:
Failure or first wrong intermediate:
Raw evidence location:

## Understanding

My explanation in plain language:
What surprised me:
One thing I still cannot explain:

## Assistance and next step

What I tried before seeking help:
Hint/tool/reference used:
What I will reconstruct without help:
Next smallest action:


<!-- Source: templates/experiment.md -->

# Experiment — descriptive name

Question:
Hypothesis:
Acceptance criterion:

## Configuration

Device/firmware:
Runtime commit/build:
Model/revision/hash:
Tokenizer/template/quant:
Backend and evidence of execution:
Input/output/context sizes:
Thread and runtime settings:
Charging, thermal and screen conditions:

## Method

Timing boundaries:
Warmup/cold terminology:
Repetitions and run order:
Controlled variables:
Known confounds:
Correctness/quality checks:

## Results

Raw-record paths:
Successful/failed run counts:
Summary statistics and sample sizes:
Memory measurement definition:
Quality result:

## Interpretation

Evidence supporting/rejecting hypothesis:
What is inferred rather than measured:
Limitations:
Next experiment:


<!-- Source: templates/weekly-review.md -->

# Week NN review

Completed days:
Carried-forward work:

## Demonstrate without help

Core concept explained:
Function reconstructed:
Failure diagnosed:

## Evidence

Working implementation:
Independent tests:
Measurement or failure report:
Ownership/numerical issue understood:

## Decision

Milestone score and reason:
Repair days needed:
Next week's first action:
One topic to defer:


<!-- Source: progress.md -->

# Daily progress

Check a day only after its acceptance check and experiment note. Carry unfinished work forward.

## Week 01: First C++ programs and an Android baseline

- [ ] Day 001: Development inventory
- [ ] Day 002: Variables and initialization
- [ ] Day 003: Functions and parameters
- [ ] Day 004: Phone development connection
- [ ] Day 005: First native baseline
- [ ] Day 006: Simple Python experiment runner
- [ ] Day 007: Week-one reconstruction
## Week 02: Control flow, numeric types and reproducible experiments

- [ ] Day 008: Conditions and validation
- [ ] Day 009: Loops and running statistics
- [ ] Day 010: Floating-point behavior
- [ ] Day 011: Integers and size arithmetic
- [ ] Day 012: Strings and configuration
- [ ] Day 013: Small baseline matrix
- [ ] Day 014: Review and measurement audit
## Week 03: References, pointers, containers and debugging

- [ ] Day 015: References and const
- [ ] Day 016: Pointers and lifetimes
- [ ] Day 017: std::vector
- [ ] Day 018: std::array and spans
- [ ] Day 019: Debugger workflow
- [ ] Day 020: Sanitizer workflow
- [ ] Day 021: Closed-book container challenge
## Week 04: Classes, RAII, moves and a minimal tensor

- [ ] Day 022: Structs and invariants
- [ ] Day 023: Constructors and RAII
- [ ] Day 024: Copy and move semantics
- [ ] Day 025: Smart pointers
- [ ] Day 026: Views and reshaping
- [ ] Day 027: Enums and errors
- [ ] Day 028: Tensor design review
## Week 05: CMake, testing, standard algorithms and checkpoint I/O

- [ ] Day 029: CMake targets
- [ ] Day 030: Testing fundamentals
- [ ] Day 031: Algorithms and lambdas
- [ ] Day 032: Templates in moderation
- [ ] Day 033: Binary checkpoint specification
- [ ] Day 034: Defensive checkpoint reader
- [ ] Day 035: Clean-build checkpoint drill
## Week 06: Concurrency, Python bindings and systems foundations

- [ ] Day 036: Threads and races
- [ ] Day 037: Mutexes and condition variables
- [ ] Day 038: Cancellation semantics
- [ ] Day 039: Python-to-C++ boundary
- [ ] Day 040: Memory mapping and page faults
- [ ] Day 041: Parallel work and false sharing
- [ ] Day 042: Foundation gate
## Week 07: Python transformer mathematics and reference operations

- [ ] Day 043: Tensor layout and linear algebra
- [ ] Day 044: Stable softmax
- [ ] Day 045: Normalization
- [ ] Day 046: Rotary positions
- [ ] Day 047: Attention reference
- [ ] Day 048: Feed-forward and residuals
- [ ] Day 049: Math reconstruction
## Week 08: Checkpoint parity, tokenization and cached generation

- [ ] Day 050: Architecture contract
- [ ] Day 051: Checkpoint loading
- [ ] Day 052: Full forward parity
- [ ] Day 053: Tokenizer and chat template
- [ ] Day 054: KV cache implementation
- [ ] Day 055: Sampling and sequence limits
- [ ] Day 056: Reference freeze
## Week 09: C++ transformer kernels and checkpoint integration

- [ ] Day 057: Matrix-vector kernel
- [ ] Day 058: RMSNorm in C++
- [ ] Day 059: RoPE in C++
- [ ] Day 060: Softmax and attention
- [ ] Day 061: Feed-forward and decoder block
- [ ] Day 062: Checkpoint-to-native bridge
- [ ] Day 063: Block parity gate
## Week 10: A complete educational C++ inference engine

- [ ] Day 064: Whole-model execution
- [ ] Day 065: Native KV cache
- [ ] Day 066: Prefill and decode paths
- [ ] Day 067: Streaming and cancellation
- [ ] Day 068: Tokenizer integration
- [ ] Day 069: Reference implementation reading
- [ ] Day 070: Engine milestone
## Week 11: Profiling and the memory hierarchy

- [ ] Day 071: Benchmark validity
- [ ] Day 072: Profile before changing code
- [ ] Day 073: Allocation discipline
- [ ] Day 074: Cache locality
- [ ] Day 075: Bandwidth and arithmetic intensity
- [ ] Day 076: Prefill versus decode workload
- [ ] Day 077: Optimization proposal
## Week 12: INT8 quantization and quality evaluation

- [ ] Day 078: Quantization arithmetic
- [ ] Day 079: Granularity
- [ ] Day 080: Native integer representation
- [ ] Day 081: Calibration
- [ ] Day 082: End-to-end quantized model
- [ ] Day 083: Task evaluation
- [ ] Day 084: Quantization report
## Week 13: INT4 packing, group quantization and fused dequantization

- [ ] Day 085: Bit operations
- [ ] Day 086: Packing arrays
- [ ] Day 087: Group quantization
- [ ] Day 088: Reference INT4 matvec
- [ ] Day 089: Fused dequantization
- [ ] Day 090: Model experiment
- [ ] Day 091: Quantization gate
## Week 14: ARM NEON, threading and measured optimization

- [ ] Day 092: Vectorization fundamentals
- [ ] Day 093: First NEON kernel
- [ ] Day 094: Quantized kernel experiment
- [ ] Day 095: Thread count and partitioning
- [ ] Day 096: False sharing and buffers
- [ ] Day 097: Fusion or reuse experiment
- [ ] Day 098: Optimization milestone
## Week 15: Android NDK deployment and native diagnostics

- [ ] Day 099: Cross-compilation fundamentals
- [ ] Day 100: Deploy the tiny engine
- [ ] Day 101: Symbols and crashes
- [ ] Day 102: Android logs and errors
- [ ] Day 103: Native profiling
- [ ] Day 104: Memory accounting
- [ ] Day 105: Device parity gate
## Week 16: Kotlin, JNI and a minimal offline application

- [ ] Day 106: Kotlin essentials
- [ ] Day 107: JNI boundary design
- [ ] Day 108: Native library integration
- [ ] Day 109: Background execution
- [ ] Day 110: Real streaming inference
- [ ] Day 111: Stop and cleanup
- [ ] Day 112: Application milestone
## Week 17: Lifecycle, sustained performance and resource policies

- [ ] Day 113: Activity recreation
- [ ] Day 114: Background and process loss
- [ ] Day 115: Model switching
- [ ] Day 116: Sustained workload design
- [ ] Day 117: Energy measurement limits
- [ ] Day 118: Resource-budget policy
- [ ] Day 119: Reliability gate
## Week 18: ExecuTorch export, graph lowering and delegation

- [ ] Day 120: Export concepts
- [ ] Day 121: Small-model deployment
- [ ] Day 122: Operator coverage
- [ ] Day 123: Quantized export
- [ ] Day 124: Delegation and fallback
- [ ] Day 125: Documented LLM attempt
- [ ] Day 126: Export milestone
## Week 19: Qualcomm NPU deployment and vendor runtime constraints

- [ ] Day 127: Choose one vendor route
- [ ] Day 128: Prepare reproducible assets
- [ ] Day 129: First target execution
- [ ] Day 130: Numerical validation
- [ ] Day 131: Prefill and decode configuration
- [ ] Day 132: Partitioning and overhead
- [ ] Day 133: Vendor path gate
## Week 20: Backend comparison and inference infrastructure

- [ ] Day 134: Upstream Snapdragon backend
- [ ] Day 135: Comparison contract
- [ ] Day 136: Automated matrix
- [ ] Day 137: Second application runtime
- [ ] Day 138: Request scheduling
- [ ] Day 139: Prefix reuse and cache validity
- [ ] Day 140: Deployment decision
## Week 21: Offline document assistant: retrieval and resource ownership

- [ ] Day 141: Capstone scope
- [ ] Day 142: Text ingestion
- [ ] Day 143: Chunking baseline
- [ ] Day 144: Embedding baseline
- [ ] Day 145: Exact retrieval
- [ ] Day 146: Answer assembly
- [ ] Day 147: RAG baseline gate
## Week 22: Capstone: memory management, quality and optional multimodality

- [ ] Day 148: Residency policy
- [ ] Day 149: Session budgeting
- [ ] Day 150: Index update integrity
- [ ] Day 151: Structured extraction task
- [ ] Day 152: Optional audio or vision branch
- [ ] Day 153: Failure injection
- [ ] Day 154: Resource-policy gate
## Week 23: Evaluation, regression testing and sustained application behavior

- [ ] Day 155: Evaluation dataset
- [ ] Day 156: Retrieval evaluation
- [ ] Day 157: Answer and citation scoring
- [ ] Day 158: Performance evaluation
- [ ] Day 159: Regression suite
- [ ] Day 160: Ablation study
- [ ] Day 161: Evaluation gate
## Week 24: Release engineering, portfolio and the next specialization

- [ ] Day 162: Model packaging
- [ ] Day 163: Interrupted installation
- [ ] Day 164: Clean build and installation
- [ ] Day 165: Soak and lifecycle release check
- [ ] Day 166: Portfolio documentation
- [ ] Day 167: Upstream contribution preparation
- [ ] Day 168: Final reconstruction and next plan
