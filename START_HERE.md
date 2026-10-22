# Start here

## The decision already made

Your weekly budget is 15–20 hours. Your aim combines inference/ML systems engineering, broad PyTorch knowledge and career readiness. The recommended plan is 39 weeks at 17 hours/week. It includes C++, GPU work and edge deployment without making Android or a particular accelerator a prerequisite for the early months.

Your existing materials describe production GenAI/backend experience and beginner-to-developing C++ depth. They also mention a local RTX 5060 Ti and phone-based experiments. These are **inherited context to verify**, not hardware or skill checks performed while assembling this roadmap.

## First session: spend at most 30 minutes deciding your starting depth

Use these checks inside Day 001, not as an extra preparation week:

| Check | Small demonstration | If it is hard |
|---|---|---|
| Python engineering | Write a function that validates a config and records a structured error | Use P03/P05 during weeks 1–2; reduce the initial runner to a recorded fixture |
| Tensor math | Explain the shapes of `[B,T,D] @ [D,V]` and a transposed matrix | Keep all layout/shape work in weeks 2–3 |
| PyTorch | Explain a parameter, buffer, gradient and `eval()` | Keep weeks 4–6 in full |
| C++ | Compile a function, explain argument lifetime and return a vector | Begin at E01; do not skip to CUDA because Python is familiar |
| Model internals | Explain why cached decoding must match a full forward pass | Keep weeks 7–9 in full |

If you already pass a check, use that session for its benchmark or source-reading task. Skip an entire week only after passing its gate with an artifact, not because the title sounds familiar. If Python and linear algebra both need substantial repair, add 2–4 foundation weeks and accept the longer calendar.

## What to do today

1. Open [Week 01](weeks/week-01.md#day-001).
2. Make a working directory named `systems-lab` somewhere convenient, separate from `originals/`. [PROJECTS.md](PROJECTS.md) suggests its structure.
3. Record your actual available hardware and already-working software. Begin on CPU if needed; don't install every runtime.
4. Complete only Day 001's task and career block. Copy [the daily log](templates/DAILY_LOG.md) into the working lab and record the result in [daily progress](trackers/daily-progress.csv).
5. Reserve five short sessions, one long session and one rest day in your own calendar.

No working lab, environment installation, model download or benchmark has been performed for you by this documentation change. The projects are your learning assignments.

## How to use a daily resource

Read the first linked local lesson for the named topic. It already contains explanations, implementation ladders and primary sources. Spend 20–30 minutes on the relevant subsection, then attempt the assigned small implementation. Use the direct external link if you need API detail. The other local references connect the same concept across curricula.

A source lesson may say “three 3-hour sessions” or “Week 18.” Those refer to its original standalone course. In this combined plan, the daily task is the required subset; the rest is reference or extension work. You are not doing three complete courses in parallel.

## Choose one baseline

Use one currently workable runtime, one small public model and one fixed input fixture for the initial serving lab. vLLM is the default source-reading lane because it matches your earlier materials; another working runtime is acceptable. Use a much smaller model or synthetic tensor fixture for reference-model and kernel work.

If setup consumes a whole session, record the first complete error and switch to a CPU or mock-fixture task. Keep a blocked hardware task in the tracker. An emulator can validate packaging/behavior, but its numbers are not phone performance.

Your first visible milestone is the [week-four report](weeks/week-04.md), not the final capstone. Career work begins in week one, with applications based on your existing experience rather than waiting for kernel expertise.
