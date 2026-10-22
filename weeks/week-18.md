# Week 18: Custom operators, dispatcher and native build boundaries

Days 120–126 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Registered CPU/native op with fake/meta behavior.

**Prerequisite:** Pass the relevant gates from [week 17](week-17.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-120"></a>

## Day 120 — Operator schema

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P35 — Module 35 — `torch.library` and Python Custom Operators](../originals/pytorch-systems-and-deployment-curriculum/modules/35-torch-library-and-python-custom-ops.md); [I41 — Module 41 — PyTorch Custom Operators](../originals/inference-engineering-curriculum/modules/41-pytorch-custom-operators.md). Focus on operator schema, using the task below to select the relevant subsection.

**External entry:** [torch.library](https://docs.pytorch.org/docs/stable/library.html). See [resource verification notes](../RESOURCES.md).

**Do:** Wrap one existing operation with torch.library using explicit shape, dtype and mutation rules.

**Done when:** Invalid inputs fail clearly and the eager path matches the oracle.

**Career block:** Recall one coding or debugging pattern relevant to custom-operator evidence; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-120.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-121"></a>

## Day 121 — Fake/meta contract

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P35 — Module 35 — `torch.library` and Python Custom Operators](../originals/pytorch-systems-and-deployment-curriculum/modules/35-torch-library-and-python-custom-ops.md). Focus on fake/meta contract, using the task below to select the relevant subsection.

**External entry:** [torch.library](https://docs.pytorch.org/docs/stable/library.html). See [resource verification notes](../RESOURCES.md).

**Do:** Implement shape propagation and run the relevant operator checks.

**Done when:** Metadata output agrees with real execution for the declared shape range.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to custom-operator evidence. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-121.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-122"></a>

## Day 122 — Native CPU extension

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P36 — Module 36 — C++/CUDA Custom Operators](../originals/pytorch-systems-and-deployment-curriculum/modules/36-cpp-cuda-custom-operators.md); [E06 — Week 06: Concurrency, Python bindings and systems foundations](../originals/edge-inference-roadmap/weeks/week-06.md). Focus on native cpu extension, using the task below to select the relevant subsection.

**External entry:** [Custom C++ and CUDA operators tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html). See [resource verification notes](../RESOURCES.md).

**Do:** Integrate the existing C++ operator into a small extension; keep the reference implementation.

**Done when:** Clean-build output agrees on supported dtypes and nontrivial shapes.

**Career block:** Add one verified custom-operator evidence observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-122.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-123"></a>

## Day 123 — Dispatcher source walk

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P37 — Module 37 — Dispatcher, ATen, and Autograd in C++](../originals/pytorch-systems-and-deployment-curriculum/modules/37-dispatcher-aten-and-autograd-in-cpp.md); [P38 — Module 38 — `native_functions.yaml`, Codegen, and the Operator Lifecycle](../originals/pytorch-systems-and-deployment-curriculum/modules/38-native-functions-codegen-and-operator-lifecycle.md). Focus on dispatcher source walk, using the task below to select the relevant subsection.

**External entry:** [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/). See [resource verification notes](../RESOURCES.md).

**Do:** Trace schema, dispatch key and implementation registration for one native operator.

**Done when:** Produce a call-path note with pinned source locations.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-123.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-124"></a>

## Day 124 — Build and ABI diagnosis

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P40 — Module 40 — Build, JIT, Ninja, CMake, and ABI Forensics](../originals/pytorch-systems-and-deployment-curriculum/modules/40-build-jit-ninja-cmake-and-abi.md); [I43 — Module 43 — Build, JIT, Ninja, CMake, and ABI](../originals/inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md). Focus on build and abi diagnosis, using the task below to select the relevant subsection.

**External entry:** [PyTorch from-source build](https://github.com/pytorch/pytorch#from-source). See [resource verification notes](../RESOURCES.md).

**Do:** Reproduce one safe configuration/build mismatch in an isolated build directory and repair it.

**Done when:** Capture the first relevant error and explain compiler versus linker responsibility.

**Career block:** Explain custom-operator evidence aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-124.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-125"></a>

## Day 125 — Custom-op gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P35 — Module 35 — `torch.library` and Python Custom Operators](../originals/pytorch-systems-and-deployment-curriculum/modules/35-torch-library-and-python-custom-ops.md); [P42 — Module 42 — Compiler Debugging, Minimization, and Regression Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/42-compiler-debugging-minimization-and-regression.md). Focus on custom-op gate, using the task below to select the relevant subsection.

**External entry:** [torch.library](https://docs.pytorch.org/docs/stable/library.html). See [resource verification notes](../RESOURCES.md).

**Do:** Test eager, compile and supported export/fake behavior for the chosen operation.

**Done when:** Document gradients or explicit inference-only limitations; no silent unsupported contract.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use custom-operator evidence as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-125.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: The operator contract survives eager and supported capture paths. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-126"></a>

## Day 126 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** The operator contract survives eager and supported capture paths.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
