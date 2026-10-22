# 32 evidence-based LinkedIn post briefs

Two optional posts per week. These are **topic briefs, not claims that the work is finished**. Publish only completed, independently reproducible and shareable work. If a result is delayed, replace the post with a precise investigation note or skip it. Do not turn this into a second content-production job.

| Week | Post A — concrete result | Post B — lesson or tradeoff | Evidence required |
|---|---|---|---|
| 1 | A public engineering question I am investigating | How I define a reproducible inference result | Scoped README and timing definitions; do not invent results |
| 2 | First small baseline with limitations | Why fresh process is not the same as cold storage | Raw records and explicit controls |
| 3 | Failed-request accounting in a benchmark | One Python test that caught an actual boundary | Code, test and failure fixture |
| 4 | Short versus long context experiment | What my first profiling attempt could not establish | Setup and observed results |
| 5 | A minimal public reproduction | How I selected a contribution scope | Reproduction and current status |
| 6 | Before/after behavior of my local patch | What a reviewer or code reading changed in my design | Diff/test; do not imply acceptance |
| 7 | Model residency measurement | A C++ lifetime bug I can now explain | Public run or isolated safe example |
| 8 | Flagship report version one | Why one tokens/s number is insufficient | Reproducible report and actual metrics |
| 9 | A synthetic table correctness fixture | Distinguishing text accuracy from table structure | Public fixture and expected output |
| 10 | Queue cancellation test | Ownership across a request lifecycle | Tested state behavior |
| 11 | Contribution follow-up or merged result | What a declined approach taught me | Accurate public status and permission where needed |
| 12 | Resource-budget decision | Why a smaller configuration won a specific workload | Quality plus latency evidence |
| 13 | A focused Android result, if ready | What source-level understanding changed | Verified target run; otherwise use host evidence |
| 14 | A controlled rerun after one change | How I avoid confusing correlation with cause | Matched configuration records |
| 15 | A concise engineering case study | One remaining limitation and next experiment | Canonical write-up and transparent limits |
| 16 | What I can now demonstrate independently | Which inference problem I will study next | Repository links and specific capability evidence |

## A reusable outline

Problem and constraint; what you tested; setup; result or observed failure; interpretation; limitation; evidence link; one genuine technical question. Keep only the fields necessary for that post. Do not force every post into identical wording.

## Example scaffold — fill with real data

'I wanted to know whether [specific change] improves [defined metric] on [actual setup]. I ran [number] repetitions with [actual workload]. The observed result was [verified value or failure]. My interpretation is [bounded claim]. This does not establish [important limitation]. Methods and raw records: [link]. The next thing I want to test is [question].'

This scaffold is for your own LinkedIn writing. Do not reuse generated prose in a repository that prohibits AI-written issue/PR text. Technical communication policies differ by project.

## Reuse without repetition

One experiment may support a result post now, a debugging lesson later and a longer case study after validation. Each should add a different useful detail. Do not repeatedly repost the same chart as several discoveries. Credit the runtime and upstream authors whose implementation you used.

## Analytics review

At weeks 4, 8 and 12 inspect which topics produced substantive engineering conversations, qualified profile interest or interview leads. Keep a post's preparation time in the tracker. If creating a visual takes longer than the experiment itself, simplify the format. There is no required follower count before applying.
