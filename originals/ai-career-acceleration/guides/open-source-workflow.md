# Open-source contribution workflow

## The objective

Become useful in one relevant subsystem. A good contribution gives maintainers a clear problem, a small understandable change and appropriate evidence. Merged work can support hiring, but maintainers do not owe reviews, endorsements or referrals. Track learning and reviewability even when merge timing is outside your control.

Choose a primary repo based on overlap with your current work and hardware. For your near-term timeline, start with vLLM Python-facing behavior, GuideLLM benchmarking, or Docling correctness/export work. Keep SGLang as a deliberate alternative. Move toward llama.cpp or ExecuTorch as native skills mature. See [repo map](repository-map.md).

## Before writing code

Read the current contribution guide, AGENTS files that apply, test instructions and AI-use policy. Inspect three to five recent merged PRs touching the same area: what counted as a sufficient test, how the description explained the behavior, and what reviewers objected to. Do not infer current conventions from a years-old tutorial.

Search issues and PRs for duplicate work, then inspect the linked discussion, assignments and recent comments. A 'good first issue' label is not proof of beginner scope or availability. A large refactor can carry that label while still needing hardware and deep context.

## Selection rubric

Score 0–2 for each: reproducible on available hardware; understandable scope; maintained subsystem; clear user impact; independent test possible; no duplicate active work; fit with your interests. Prefer a high score and small change. Reject work that requires inaccessible multi-GPU hardware unless a maintainer confirms a practical validation path.

For a feature, seek the project's required agreement or RFC first. For a bug, reproduce it and identify a failing regression test. A locally useful experiment belongs in your repository until it is suitable for upstream. Do not create an upstream issue solely to advertise your project.

## The 12-session sprint

| Session | Work | Concrete output |
|---|---|---|
| 1 | Read policies and inspect merged changes | Policy and test notes |
| 2 | Build/install the relevant development path | Minimal working environment |
| 3 | Trace one request or utility path | Call-path note |
| 4 | Inspect 3–5 candidate issues/observed gaps | Selection rubric |
| 5 | Check duplicates and reproduce one candidate | Exact command and failure |
| 6 | Write an independent failing check | Red regression evidence |
| 7 | Propose/confirm scope if project requires it | Human-authored scope note |
| 8 | Make the smallest coherent change | Focused patch |
| 9 | Run targeted tests and relevant integration checks | Honest test record |
| 10 | Review the diff and update necessary docs | Explained lines and clean scope |
| 11 | Prepare submission according to project policy | Reviewable contribution |
| 12 | Respond to feedback or document blocked status | Updated tests and lessons |

Each session is a work block, not necessarily a day. A repo setup can take longer. Do not rush into a large patch to keep a calendar promise.

## Testing and evidence

Record what you ran, on which hardware and version, and what you could not run. A unit test may validate logic but not a GPU performance claim. A benchmark must preserve correctness and state workload conditions. CI access may be gated; follow project rules and do not repeatedly request expensive runs before addressing local failures.

Documentation can be valuable when it corrects a real installation or behavior gap demonstrated by a reproduction. Trivial formatting and typo churn is not a useful visibility strategy and is explicitly discouraged by some projects. Do not fabricate a bug to create a contribution.

## Submission and review

Keep one concern per change. Explain the user-visible trigger and final behavior, not a long chat history. Do not bundle unrelated refactors. Ask a precise question when uncertain. Follow the current CLA/DCO/sign-off process and understand what you are certifying.

Answer reviewer comments with the technical change and test evidence. If a suggestion is unclear, ask for clarification rather than guessing. Do not tag many maintainers or repeatedly bump. Follow up once after a reasonable project-specific interval and substantive update; then respect capacity. Close your work gracefully if superseded.

## Important repository-specific policy snapshot

vLLM's inspected guide requires DCO sign-off and describes disclosure/attribution for nontrivial AI-assisted code [O1]. llama.cpp's inspected guide requires AI-use disclosure, limits new contributors to one open PR, and prohibits AI-written issue/PR/discussion text [O3]. Therefore this pack supplies a preparation checklist rather than ready-to-submit GitHub prose for llama.cpp. You should write those posts yourself, read the current policy and explain every submitted line. These are current project rules, not universal rules for all repositories.

SGLang documents test placement, validation and gated CI [O2]. Exact commands and policy can change. Read the guide in the checkout you use; your roadmap is not a substitute for it.

## Turning work into career evidence

After a public reviewable milestone, write what you investigated, what the test catches, the change, validation and status. Distinguish proposed, submitted, accepted and merged. Credit maintainers and collaborators. A closed PR can still demonstrate judgment if you learned a real architectural constraint, but do not describe it as an accepted contribution.

Do not attach job requests to technical issue threads. If a professional relationship develops, keep career conversation separate and appropriate.

## Sources

[O1] https://docs.vllm.ai/en/latest/contributing/

[O2] https://docs.sglang.io/docs/developer_guide/contribution_guide

[O3] https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md
