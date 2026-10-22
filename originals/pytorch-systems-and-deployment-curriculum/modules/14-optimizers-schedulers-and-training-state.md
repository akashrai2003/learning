# Module 14 — Optimizers, Schedulers, and Training State

> **Phase:** I — Engineering foundations  
> **Time:** 9–11 focused hours across three sessions  
> **Prerequisites:** Modules 10–13, gradient descent, and basic probability/statistics

## Existing inference curriculum bridge

This is the training-side complement to [Module 08 — Normalization, Residuals, and Stability](../../inference-engineering-curriculum/modules/08-normalization-residuals-and-stability.md), [Module 18 — Quantization Fundamentals](../../inference-engineering-curriculum/modules/18-quantization-fundamentals.md), and [Module 25 — FP8 Training and Inference](../../inference-engineering-curriculum/modules/25-fp8-training-and-inference.md). Inference systems inherit the weights, precision policy, and model behavior created by these update rules. Later modules cover exact serialization and AMP; this module establishes the core optimizer contract first.

## Why this matters and outcomes

An optimizer is stateful numerical code, not a one-line appendage to loss.backward(). Its update depends on current gradients, parameter groups, momentum/variance buffers, learning-rate schedule, step count, clipping policy, and often mixed-precision state. A training run can look healthy while silently omitting a parameter group, decaying a parameter unintentionally, stepping a scheduler at the wrong cadence, or resuming with mismatched state.

By the end, you can derive and test SGD/momentum and AdamW updates; create deliberate parameter groups; choose and verify scheduler cadence; handle gradient accumulation and clipping correctly; inspect optimizer state; and define the minimum state needed for a faithful resume.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P14 — Optimizers, schedulers, and training state**.

1. [torch.optim](https://docs.pytorch.org/docs/stable/optim.html) — learn the API or system contract.
2. [Adam paper](https://arxiv.org/abs/1412.6980) — study the underlying design, paper, or specification.
3. [Learning-rate scheduler API](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch examples](https://github.com/pytorch/examples) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch recipes](https://docs.pytorch.org/tutorials/recipes/recipes_index.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

For SGD with momentum, given gradient g_t and velocity v_t, a common form is:

v_t = μ v_(t-1) + g_t  
θ_(t+1) = θ_t - η v_t.

Exact variants differ in dampening, Nesterov handling, and weight decay, so read the optimizer’s documented formula rather than assume the name fixes all details. AdamW maintains first and second moment estimates:

m_t = β1 m_(t-1) + (1-β1) g_t  
v_t = β2 v_(t-1) + (1-β2) g_t²  
θ_(t+1) = θ_t - η [m_hat_t / (sqrt(v_hat_t)+ε) + λ θ_t].

The last term is decoupled weight decay: it is not merely adding L2 penalty into the adaptive gradient. Optimizer state can exceed parameter payload for adaptive methods, a material systems cost during fine-tuning and distributed training.

Parameter groups encode update policy. For example, a project may give embeddings, norms, bias, adapters, or weight matrices different learning rates or decay choices. This is a model-specific policy, not a cargo-cult list. Generate groups by named parameters, assert every trainable parameter occurs exactly once, record names/counts/bytes, and test a one-step update. A frozen parameter should have requires_grad disabled or be absent from optimizer groups according to an explicit policy.

A normal step sequence is: set mode; forward; loss; backward; optionally unscale and clip; optimizer.step; scheduler.step when the scheduler’s documented cadence says so; then clear gradients for the next window. Under gradient accumulation, divide or otherwise normalize loss consistently, call backward across microbatches, and step only at the logical update boundary. Gradient clipping is applied to the accumulated gradient after any AMP unscale and before optimizer.step. Module 22 adds the exact autocast/scaler path.

Schedulers change optimizer group learning rates; they do not own the optimization state. Some step per update, some per epoch, and metric-driven schedulers consume a validation metric. Record a short LR trace from the first steps and test it. Off-by-one schedule errors can waste a run while every API call appears valid.

A faithful resume is larger than model parameters: it needs model state, optimizer state, scheduler state, scaler state if used, RNG/data-sampler state, global step/epoch/microstep, configuration, and dataset/model revisions. Module 16 will implement secure, versioned save/load behavior; define the state inventory here so you know what “resume” means.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/training/optimizer_oracles.py. Implement one-step SGD/momentum and AdamW equations in float64 and compare them to PyTorch on fixed scalar/vector parameters.
2. Write build_param_groups(model) from named_parameters. Emit a manifest with each parameter name, shape, group, LR, decay, and duplicate/missing checks.

**Optimized**

3. Build train_step with explicit gradient-accumulation count, finite-gradient check, optional norm clipping, optimizer step, scheduler step, and zeroing policy. Return structured metrics, not just loss.
4. Add tests for frozen parameters, tied parameters, empty/None gradients, two parameter groups, clipping threshold, and a scheduler’s first ten learning rates.

**Integrated**

5. Make a deliberately incomplete checkpoint inventory and show which resumed trajectories diverge; defer the actual serialization implementation to Module 16.

## Three focused sessions

1. **Update equations (3h):** calculate two updates by hand, implement an FP64 oracle, compare SGD/momentum/AdamW, and inspect created optimizer buffers.
2. **Groups and schedules (3h):** construct named groups, prove coverage, plot the first learning-rate steps, and create one intentional cadence bug.
3. **Training state (3h):** implement microbatch accumulation and clipping; record every mutable state transition in a single training step and design a resume-equivalence test plan.

## Experiments and deliverables

Commit optimizer_oracles.py, parameter-group manifest, scheduler trace, train-step tests, and training-state inventory. Include model/optimizer/scheduler memory footprints and a table explaining why every state component is needed for a numerical resume. Run a fixed tiny problem where a wrong group or schedule visibly changes the expected update.

## Common failure modes

- Creating an optimizer before all intended parameters are registered.
- Adding a parameter twice or omitting a trainable parameter from every group.
- Applying an undocumented no-decay rule without recording it.
- Stepping a scheduler at the wrong unit or in the wrong order for its API.
- Calling optimizer.step at each microbatch during intended accumulation.
- Clipping before AMP unscale, or treating a skipped/non-finite step as a successful update.
- Saving only model weights and calling the next run a faithful resume.

## Mastery gate

Given a tiny model and an update trace, derive the next parameter, momentum, and AdamW state exactly. Produce a group manifest with total coverage and no duplicates, a tested learning-rate trace, and a state inventory sufficient to explain exact versus approximate resume. Diagnose a deliberately introduced accumulation or scheduler-cadence bug.

## Primary sources and canonical code paths

- [PyTorch optimizers](https://docs.pytorch.org/docs/stable/optim.html)
- [PyTorch AdamW API](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch learning-rate schedulers](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [PyTorch clip_grad_norm](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- Inspect [optimizer.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/optimizer.py), [adamw.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py), [lr_scheduler.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py), and optimizer tests.

## OSS bridge and next

Strong contribution candidates are a precise scheduler regression, parameter-group validation helper, documentation correction with a runnable trace, or an optimizer state compatibility test. Avoid benchmark-only changes without a numerical oracle. Module 15 adds reproducibility and random-state control, then Module 16 turns the state inventory into safe, versioned checkpointing.
