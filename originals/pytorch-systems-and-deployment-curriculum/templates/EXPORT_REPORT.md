# Export and Deployment Report

## Source program contract

- Model/checkpoint/preprocessing revision:
- Eager reference command:
- Inputs, dynamic dimensions, range constraints, and non-tensor constants:
- Mutation/state/control-flow assumptions:

## Artifact chain

| Stage | Tool/version | Artifact path/hash | Accepted shapes | Unsupported/fallback behavior |
|---|---|---|---|---|
| Eager | | | | |
| `torch.export` | | | | |
| ONNX / AOT / TensorRT / ExecuTorch | | | | |
| Target runtime | | | | |

## Parity matrix

| Input class | Eager vs export | Export vs runtime | Task metric | Tolerance / result |
|---|---:|---:|---:|---|
| Nominal | | | | |
| Minimum dynamic | | | | |
| Maximum dynamic | | | | |
| Adversarial / edge | | | | |

## Runtime evidence

- Device/runtime/driver/provider/delegate versions:
- Build/profile/tactic/partition options:
- Artifact size, compile/export time, load time, warm latency/throughput:
- Peak host/device memory and power/thermal notes if applicable:
- Inspect graph partitions, execution provider assignment, or delegate calls:

## Conclusion

State exactly what the artifact is valid for, where it falls back or fails, and
the next decision it supports. Do not call an export “portable” without naming
the runtime and input contract.
