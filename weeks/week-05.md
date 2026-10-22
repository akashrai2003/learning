# Week 05: Checkpoints, CMake and model-state discipline

Days 029–035 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Reloadable model state and a clean native build.

**Prerequisite:** Pass the relevant gates from [week 04](week-04.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-029"></a>

## Day 029 — Serialization contract

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P16 — Module 16 — Serialization, Checkpoints, and Model Artifacts](../originals/pytorch-systems-and-deployment-curriculum/modules/16-serialization-checkpoints-and-model-artifacts.md); [P11 — Module 11 — Modules, Parameters, Buffers, Hooks, and State](../originals/pytorch-systems-and-deployment-curriculum/modules/11-modules-parameters-buffers-hooks-and-state.md). Focus on serialization contract, using the task below to select the relevant subsection.

**External entry:** [Serialization note](https://docs.pytorch.org/docs/stable/notes/serialization.html). See [resource verification notes](../RESOURCES.md).

**Do:** Save and reload tiny-model parameters and buffers; record model config, format and revision.

**Done when:** Reloaded outputs match the reference for fixed inputs.

**Career block:** Practice a Python data-structure problem with tests. Use the [career reference](../originals/ai-career-acceleration/weeks/week-05.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-029.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-030"></a>

## Day 030 — CMake targets

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E05 — Week 05: CMake, testing, standard algorithms and checkpoint I/O](../originals/edge-inference-roadmap/weeks/week-05.md). Focus on cmake targets, using the task below to select the relevant subsection.

**External entry:** [CMake tutorial: executable and library targets](https://cmake.org/cmake/help/latest/guide/tutorial/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Move the native buffer and dot product into a library and executable using CMake.

**Done when:** A clean build directory produces the same test executable.

**Career block:** Act on the next application or existing response. Use the [career reference](../originals/ai-career-acceleration/weeks/week-05.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-030.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-031"></a>

## Day 031 — Training-state replay

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P15 — Module 15 — Randomness, Determinism, and Reproducibility](../originals/pytorch-systems-and-deployment-curriculum/modules/15-randomness-determinism-and-reproducibility.md); [P14 — Module 14 — Optimizers, Schedulers, and Training State](../originals/pytorch-systems-and-deployment-curriculum/modules/14-optimizers-schedulers-and-training-state.md). Focus on training-state replay, using the task below to select the relevant subsection.

**External entry:** [Reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html). See [resource verification notes](../RESOURCES.md).

**Do:** Save optimizer state and random-state information for a tiny training run; resume after a checkpoint.

**Done when:** Compare the resumed next step with an uninterrupted controlled run.

**Career block:** Improve one README section based on the report's reproducibility gaps. Use the [career reference](../originals/ai-career-acceleration/weeks/week-05.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-031.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-032"></a>

## Day 032 — Native checkpoint reader

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E05 — Week 05: CMake, testing, standard algorithms and checkpoint I/O](../originals/edge-inference-roadmap/weeks/week-05.md); [P16 — Module 16 — Serialization, Checkpoints, and Model Artifacts](../originals/pytorch-systems-and-deployment-curriculum/modules/16-serialization-checkpoints-and-model-artifacts.md). Focus on native checkpoint reader, using the task below to select the relevant subsection.

**External entry:** [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Define a tiny educational tensor file with shape, dtype and payload length; write it in Python and read it in C++.

**Done when:** Reject truncation, wrong shape and unsupported dtype before accessing memory.

**Career block:** Write a failing fixture for the selected contribution if one is justified. Use the [career reference](../originals/ai-career-acceleration/weeks/week-05.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-032.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-033"></a>

## Day 033 — Regression discipline

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P05 — Module 05 — Testing, Debugging, and Benchmark Discipline](../originals/pytorch-systems-and-deployment-curriculum/modules/05-testing-debugging-and-benchmark-discipline.md); [P60 — Module 60 — PyTorch Testing, CI, and Triage](../originals/pytorch-systems-and-deployment-curriculum/modules/60-pytorch-testing-ci-and-triage.md). Focus on regression discipline, using the task below to select the relevant subsection.

**External entry:** [PyTorch testing overview](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#testing). See [resource verification notes](../RESOURCES.md).

**Do:** Write a regression fixture for one actual parser or checkpoint failure; run the smallest relevant test set.

**Done when:** The test fails on the broken behavior and passes on the repair.

**Career block:** Rehearse a production ownership story with honest metrics. Use the [career reference](../originals/ai-career-acceleration/weeks/week-05.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-033.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-034"></a>

## Day 034 — Build dossier

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [E05 — Week 05: CMake, testing, standard algorithms and checkpoint I/O](../originals/edge-inference-roadmap/weeks/week-05.md); [I43 — Module 43 — Build, JIT, Ninja, CMake, and ABI](../originals/inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md). Focus on build dossier, using the task below to select the relevant subsection.

**External entry:** [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Document compiler, build flags, dependencies and the exact clean-build command.

**Done when:** A second directory can rebuild without copying cached binaries.

**Career block:** Do a mock plus focused contribution work; carry unfinished scope instead of opening another issue. Use the [career reference](../originals/ai-career-acceleration/weeks/week-05.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-034.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Reproduce outputs after reload and reject malformed artifacts. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-035"></a>

## Day 035 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Reproduce outputs after reload and reject malformed artifacts.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
