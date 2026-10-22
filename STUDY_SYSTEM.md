# Study system and pacing

## One budget, six working days

Use Day 1–6 on the days that suit your job. “Weekday” below is just a convenient label; no start date is assumed.

| Session | Weeks 1–16 | Weeks 17–39 |
|---|---|---|
| Five short sessions | 120 min technical + 30 min career = 150 min each | 135 min technical + 15 min career = 150 min each |
| One long session | 120 min technical + 90 min career + 60 min review = 270 min | 165 min technical + 45 min career + 60 min review = 270 min |
| Rest | 0 required minutes | 0 required minutes |
| Weekly total | 1,020 min = 17 h | 1,020 min = 17 h |

Split the long session into two blocks. If that day is unavailable, distribute its work across other days while preserving one rest day and your weekly ceiling.

A 120-minute technical block: 10 recall, 25 reading, 60 implementation/source inspection, 20 verification, 5 logging. For 135 minutes, add 15 implementation minutes; for 165, add 30 implementation and 15 verification minutes. The weekly review is separate and already budgeted.

## Make the task small enough to finish

Each daily assignment is a minimum slice. For a difficult implementation, first finish one shape, one dtype and one positive/negative case. Expand only after those are understood. Use the existing oracle, fixtures and report schema rather than starting a new repository each time.

End every working day with an artifact: code, a failing/passing test, raw result, source path, derivation or a precise blocker. Reading alone can be valid for a survey day, but it should produce an explanation or decision note.

## Recall and independence

Before opening the lesson, explain yesterday's mechanism or reconstruct a small function. Revisit one idea approximately 3, 7 and 21 days later inside the existing recall block. Keep a small queue in the daily log; do not create a separate flashcard course.

Attempt the core implementation yourself before consulting a complete solution. If you use an assistant, record the hint and reconstruct the key idea later without it. Ask for a counterexample, explanation or review when that helps more than a finished implementation.

## Completion and repair

Use statuses `not_started`, `in_progress`, `done`, `blocked`, `deferred` in the daily tracker. Rest rows can remain `rest`. A study row becomes `done` only when its completion check has evidence. Missing hardware can leave one task blocked while a labeled host substitute is completed; it does not verify the missing hardware.

For an unexplained numerical, ownership or state bug, pause dependent tasks. For a known environment limitation, time-box diagnosis to one session, record versions/commands/error and continue a compatible CPU or fixture route. Revisit at the buffer or when the environment changes.

Buffers at weeks 12, 24 and 36 reserve time for difficult tasks, interviews, illness and clean reproduction. If they are not needed, repeat a closed-book gate or take a lighter week. Do not fill them with additional frameworks.

## Busy-week mode

If only 6–8 hours are available, complete two technical blocks, one coding/design practice, one meaningful career next action and a short review. Pause public posting and optional sources. Resume at the next unfinished session; do not try to make up seven days in one weekend. Record the actual date so progress is not confused with the planned calendar.

At a consistent 15-hour budget, stretch the schedule to roughly 45 weeks. At 20 hours, spend the extra three hours on the current gate or review response; do not automatically add another specialization.

## Career-priority mode

The original career plan aimed at a near-term role change and gave career activity much more time. If interviews or a job transition are currently urgent, use **8 technical + 8 career + 1 review hour** for the affected weeks, with the same 17-hour total. Reduce new technical scope and shift later gates. This is an explicit change in priority, not invisible overtime.

An actual interview includes preparation and debrief within the career allocation. Reduce content and new OSS scope first. In the initial four-hour lane, a reasonable week is one reviewed coding exercise, one mock/design block, one or two carefully screened applications/next actions and a small evidence update. Quality and fit determine the volume; none is a guaranteed output quota.

After week 16, the two-hour maintenance lane is appropriate only if job-search urgency is lower. Continue career-priority mode if it is not. Use [active trackers](trackers/README.md), not the archived copies.

## Keep measurement honest

For every experiment record hardware, software/model revisions, inputs, shape distribution, precision, warm-up, synchronization, repetitions and raw results. Define timing boundaries and memory metrics. Separate startup, prefill, decode, throughput and quality; include failures. A slow but understood implementation is a valid result.

Use [the experiment template](templates/EXPERIMENT.md) and the original [inference benchmarking guide](originals/inference-engineering-curriculum/BENCHMARKING.md). No paid hardware is required to begin. Multi-GPU, phone and vendor-accelerator observations remain conditional on actual access and compatible software.
