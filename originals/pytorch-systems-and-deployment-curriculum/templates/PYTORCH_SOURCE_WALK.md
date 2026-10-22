# PyTorch Source Walk

## Question and revision

- Question being answered:
- Repository / commit / branch:
- Public API or user-visible behavior:
- Reproducer or test command:

## Call path

| Layer | File / symbol | Input-output contract | Key invariant / uncertainty |
|---|---|---|---|
| User API | | | |
| Python wrapper / graph capture | | | |
| Schema / dispatcher | | | |
| Autograd / FakeTensor / meta | | | |
| Native / generated / kernel code | | | |
| Test / CI target | | | |

## Execution and state diagram

Draw the data/control path, including device/dtype/layout, mutation/aliasing,
shape constraints, dispatch keys, and any generated artifact boundary.

## Experiments

- Confirmed behavior:
- Counterexample/failure:
- Relevant logs/traces:
- What remains unknown:

## Contribution implication

Name the smallest plausible test/documentation/fix location, compatibility risk,
and the maintainer question you would ask before making a broad change.
