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
