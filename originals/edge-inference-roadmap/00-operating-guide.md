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
