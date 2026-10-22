# Milestones and honest self-assessment

Score each area 0–3: **0** not attempted; **1** runs with substantial help or unexplained behavior; **2** independently implemented and tested; **3** independently explained, reconstructed and investigated under failures. A total is less important than unresolved zeros in prerequisites.

## Week 6: Native foundations

Required evidence: a clean host build, Tensor/Shape contracts, validated binary reader, independent fixtures, sanitizer-based debugging and a bounded worker queue. Explain a borrowed view's lifetime, a move, integer overflow and worker shutdown. If you cannot do this, spend another one or two weeks repairing foundations before the native transformer.

## Week 10: Inference understanding

Required evidence: Python reference, full C++ forward pass, intermediate parity, working KV cache, bounded generation and a small checkpoint. Explain each tensor dimension, why prefill differs from decode, and how cache positions affect RoPE. An attractive chat demo without parity evidence does not pass.

## Week 14: Optimization discipline

Required evidence: valid release benchmarks, profile, INT8 and INT4 references, at least one ARM optimization attempt, and preserved correctness. Explain why one optimization failed. You can pass with modest speedups if the explanation and measurements are sound.

## Week 17: Android reliability

Required evidence: native deployment, minimal app, working cancellation, coherent lifecycle handling, diagnostics and sustained measurements. Explain the callback lifetime and model/session ownership. A crash or deadlock on repeated start/stop requires repair before the capstone.

## Week 20: Backend investigation

Required evidence: one verified deployment path, one export or accelerator investigation, output validation and a versioned compatibility matrix. A well-documented NPU incompatibility can pass if CPU deployment works and your conclusions are bounded by evidence. Do not reward unsupported hardware claims.

## Week 24: Integrated engineering

Required evidence: offline application, documented on-device stages, fixed evaluation set, resource policies, failure injection, clean-build instructions and release metadata. Explain why the chosen model/backend is appropriate and what the application cannot do reliably.

## Oral questions

1. Which allocations scale with parameters, context, batch and concurrent sessions?
2. What is stored in the KV cache, and what makes it invalid for reuse?
3. How would you distinguish a tokenizer bug from a matrix kernel bug?
4. Why might an INT4 model be slower than INT8?
5. When does a vector invalidate a TensorView?
6. How do you cancel a worker without freeing memory it still uses?
7. What proves the expensive graph actually executes on the NPU?
8. Why is process restart not a guaranteed cold-storage benchmark?
9. What do your memory measurements count and potentially overlap?
10. Which quality failures are caused by retrieval versus generation?

## Portfolio completion rubric

For each of the three projects score correctness, ownership, reproducibility, measurement, failure handling and explanation. Aim for level 2 across all areas and level 3 in selected areas. Identify the weakest demonstrated area and use it to choose the next specialization. Do not confuse a fixed calendar with a guarantee of production mastery.
