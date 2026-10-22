# C++ from the beginning to inference-oriented engineering

## How this differs from a general C++ course

You will learn the language through arrays, model files, kernels and request processing. That still requires ordinary language fundamentals: skipping types, initialization, object lifetime or error handling because you already know Python produces fragile native code.

Use LearnCpp's index to locate the named topics. Do not assume Python reference semantics, integer behavior or memory safety carry over. C++ gives you both useful control and more ways to make an invalid program.

## Stages and evidence

| Stage | Required concepts | Evidence |
|---|---|---|
| Weeks 1–2 | Compilation/linking, initialization, functions, types, conditions, loops, strings, numeric limits | A robust CLI and numeric estimator |
| Week 3 | References, pointers, const, containers, spans, lifetime, debugging | Correct array operations and diagnosed memory errors |
| Week 4 | Structs/classes, constructors, destructors, invariants, RAII, copies, moves, unique ownership | Tensor and TensorView with explicit contracts |
| Week 5 | CMake, unit tests, algorithms, lambdas, basic templates, binary I/O | Validated checkpoint loader |
| Week 6 | Threads, mutexes, condition variables, cancellation, mappings, Python bindings | Bounded executor and reference parity |
| Weeks 7–10 | Interfaces, integration, numerical implementation and ownership across modules | Small native transformer |
| Weeks 11–14 | Allocation behavior, layout, vectorization, intrinsics, profiling | Evidence-backed optimization |
| Weeks 15–24 | ABI, shared libraries, JNI, lifecycle, deployment and release contracts | Reliable Android integration |

## The essential mental models

**Object versus storage:** storage can exist without a live object of the type you want. Pointer casting does not establish alignment, lifetime or valid representation. For binary checkpoints, decode fields explicitly; do not assume mapped bytes can always be reinterpreted as arbitrary C++ objects.

**Ownership versus access:** a vector may own memory while a span borrows it. A function accepting a view must document that the owner remains alive and stable. Vector reallocation can invalidate a view even when the vector object itself remains alive.

**Value versus identity:** copying a Tensor can copy its data if it has value semantics. Sharing immutable weights can instead be appropriate at the model layer. Make these decisions explicit; a smart pointer is not a substitute for an ownership design.

**Const versus concurrency:** a const method can still touch shared external state or mutable members. Thread safety follows synchronization and ownership rules, not spelling alone.

**Move versus copy:** std::move expresses permission to move through overload resolution; it does not itself transfer data. Do not move from a resource and then assume its previous contents remain accessible. Prefer standard containers and the rule of zero before writing custom special member functions.

**Exceptions versus boundaries:** RAII should clean resources during failures. At a C or JNI boundary, translate exceptions into an explicit error contract. Do not let a native exception escape through an interface that cannot handle it.

## Topics to add when motivated

Basic inheritance and virtual functions are useful for a backend interface, but you do not need a deep hierarchy for Tensor. Compare a virtual interface, a tagged variant and a plain function table on clarity before worrying about nanoseconds. Study templates only far enough to understand the generic code you use; advanced expression templates are optional.

Learn std::optional and std::variant when designing missing values or tagged results. std::expected is a later-standard option; do not introduce it into C++20 code without a deliberate compatible implementation. Read atomics and the memory model before designing a lock-free queue. A mutex-based bounded queue is sufficient for the main course.

## Recommended learning checks

- Explain which objects exist at every line of a function returning a view.
- Explain why a signed integer overflow is not a valid wraparound technique.
- Predict when a vector append invalidates an address.
- Distinguish a link error from a compile error and a loader error.
- Diagnose an out-of-bounds access using tools rather than adding random prints.
- Explain how a waiting worker exits during shutdown.
- Explain whether a Tensor copy is intended, expensive or forbidden.
- Review a binary loader as if its input were corrupted.

## Definition of production progress

After six months, a reasonable result is well-structured small native components, good debugging discipline and a credible Android capstone. Large-scale production C++ expertise still requires code review, maintenance and real failure exposure. Use the rubric to show evidence instead of awarding yourself a title after a fixed number of days.
