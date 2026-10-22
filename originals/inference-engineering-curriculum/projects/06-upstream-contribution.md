# Capstone 06 — Upstream-Quality Contribution

## Objective

Produce one contribution that maintainers can evaluate without reconstructing
your environment or intent. A merged PR is ideal but not fully under your
control; the grading target is upstream quality and responsible collaboration.

## Candidate scope

Preferred first scopes:

- minimal reproducer plus regression test for a confirmed bug;
- missing shape/dtype/backend correctness case;
- misleading fallback or build diagnostic;
- model loader/config/tokenizer edge case;
- benchmark or profiler instrumentation fix;
- documentation derived from a verified source walk;
- localized kernel regression on hardware you own.

Avoid an unsolicited engine redesign, new public API, or new quantization format
as the first contribution.

## Phase 1 — reconnaissance

Before editing:

- read current contribution and AI-use policies;
- build/test the pinned main branch;
- search issues, PRs, discussions, roadmaps, and commit history;
- identify subsystem ownership;
- reproduce with a minimal input;
- determine whether the behavior is a bug, unsupported case, or expected
  fallback;
- ask for alignment when project policy or scope warrants it.

Write communications yourself in accordance with project rules. You must
understand and be able to defend every submitted line.

## Phase 2 — evidence

Create:

- sanitized environment manifest;
- exact reproduction command;
- smallest failing test;
- error/backtrace/profile;
- suspected call path;
- last-known-good or affected-version information if discoverable;
- scope matrix: affected and unaffected shapes/hardware/models.

Do not begin with a large patch that hides the original failure.

## Phase 3 — design

State:

- invariant to restore or performance mechanism to change;
- why this layer owns the fix;
- alternatives considered;
- compatibility and fallback;
- test plan;
- benchmark/quality plan;
- documentation and release implications;
- how the patch can be split.

Seek maintainer agreement before high-cost implementation.

## Phase 4 — implementation

Make reviewable commits:

1. reproducer/test when appropriate;
2. minimal semantic fix;
3. optimization or additional backend;
4. documentation/benchmark updates.

Follow formatting, static checks, commit-signoff, PR title, and CI rules exactly.
Do not refactor nearby code unless necessary.

## Phase 5 — validation

Required:

- targeted tests;
- prescribed project lint/static checks;
- relevant larger test suite;
- failure-path/fallback tests;
- performance comparison if the critical path changes;
- quality/perplexity/KL checks if numerics or quantization change;
- cross-platform analysis and tests available to you;
- clean build/environment reproduction.

State what you could not test and why.

## Phase 6 — review

Respond with evidence and technical reasoning. When requested changes expand the
scope, restate trade-offs rather than silently growing the patch. Update tests
with bug discoveries. Rebase according to project convention. Do not pressure
maintainers for timelines.

After merge—or if declined—write a retrospective:

- what assumption was wrong;
- which repository invariant mattered;
- what review feedback generalized;
- what follow-up belongs upstream versus locally;
- whether benchmark claims survived independent review.

## Acceptance gate

The dossier must let another engineer:

- reproduce the pre-patch behavior;
- understand the intended contract;
- run tests that detect regression;
- inspect a narrowly scoped patch;
- reproduce performance/quality evidence;
- see limitations and unsupported cases;
- verify compliance with current project policies.

If the PR is not opened or merged, include a patch series and mock review from
another engineer. Do not misrepresent local work as accepted upstream work.

