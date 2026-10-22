# Week 30: Android NDK, JNI and native execution

Days 204–210 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** On-device native baseline or documented host fallback.

**Prerequisite:** Pass the relevant gates from [week 29](week-29.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-204"></a>

## Day 204 — Device and ABI

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E15 — Week 15: Android NDK deployment and native diagnostics](../originals/edge-inference-roadmap/weeks/week-15.md); [E01 — Week 01: First C++ programs and an Android baseline](../originals/edge-inference-roadmap/weeks/week-01.md). Focus on device and abi, using the task below to select the relevant subsection.

**External entry:** [Android NDK guide](https://developer.android.com/ndk/guides). See [resource verification notes](../RESOURCES.md).

**Do:** Record an available phone's OS, ABI, real RAM and storage; configure NDK/ADB or document missing hardware.

**Done when:** A small native executable runs on the actual target or host fallback is explicitly labeled.

**Career block:** Recall one coding or debugging pattern relevant to native ownership; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-204.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-205"></a>

## Day 205 — Cross-build

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E15 — Week 15: Android NDK deployment and native diagnostics](../originals/edge-inference-roadmap/weeks/week-15.md); [E05 — Week 05: CMake, testing, standard algorithms and checkpoint I/O](../originals/edge-inference-roadmap/weeks/week-05.md). Focus on cross-build, using the task below to select the relevant subsection.

**External entry:** [Android NDK guide](https://developer.android.com/ndk/guides). See [resource verification notes](../RESOURCES.md).

**Do:** Build the smallest native program/library using the target toolchain and existing CMake skills.

**Done when:** Host binaries are not mistaken for target binaries.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to native ownership. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-205.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-206"></a>

## Day 206 — JNI boundary

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E16 — Week 16: Kotlin, JNI and a minimal offline application](../originals/edge-inference-roadmap/weeks/week-16.md). Focus on jni boundary, using the task below to select the relevant subsection.

**External entry:** [Android JNI tips](https://developer.android.com/training/articles/perf-jni). See [resource verification notes](../RESOURCES.md).

**Do:** Expose one tiny native operation to a minimal Android UI; learn only the Kotlin lifecycle basics needed.

**Done when:** Check argument errors and native exceptions without leaking references.

**Career block:** Add one verified native ownership observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-206.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-207"></a>

## Day 207 — Native ownership

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E16 — Week 16: Kotlin, JNI and a minimal offline application](../originals/edge-inference-roadmap/weeks/week-16.md); [P39 — Module 39 — LibTorch and the C++ Frontend](../originals/pytorch-systems-and-deployment-curriculum/modules/39-libtorch-and-cpp-frontend.md). Focus on native ownership, using the task below to select the relevant subsection.

**External entry:** [Android JNI tips](https://developer.android.com/training/articles/perf-jni). See [resource verification notes](../RESOURCES.md).

**Do:** Define who owns model, session and output buffers across JNI and worker threads.

**Done when:** Repeated create/run/destroy leaves no growing owned-state count.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-207.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-208"></a>

## Day 208 — Model baseline

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E15 — Week 15: Android NDK deployment and native diagnostics](../originals/edge-inference-roadmap/weeks/week-15.md); [E18 — Week 18: ExecuTorch export, graph lowering and delegation](../originals/edge-inference-roadmap/weeks/week-18.md). Focus on model baseline, using the task below to select the relevant subsection.

**External entry:** [Android NDK guide](https://developer.android.com/ndk/guides). See [resource verification notes](../RESOURCES.md).

**Do:** Run one supported small model through the established native/edge runtime on the device.

**Done when:** Record model, runtime, ABI and first-run versus warm timing.

**Career block:** Explain native ownership aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-208.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-209"></a>

## Day 209 — Mobile gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [E16 — Week 16: Kotlin, JNI and a minimal offline application](../originals/edge-inference-roadmap/weeks/week-16.md); [E17 — Week 17: Lifecycle, sustained performance and resource policies](../originals/edge-inference-roadmap/weeks/week-17.md). Focus on mobile gate, using the task below to select the relevant subsection.

**External entry:** [Android JNI tips](https://developer.android.com/training/articles/perf-jni). See [resource verification notes](../RESOURCES.md).

**Do:** Demonstrate bounded generation or the small supported model with a cancellation/error path.

**Done when:** If no device exists, produce host parity and a precise unverified-device checklist.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use native ownership as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-209.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain ABI, JNI lifetime and cross-language error handling. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-210"></a>

## Day 210 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain ABI, JNI lifetime and cross-language error handling.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
