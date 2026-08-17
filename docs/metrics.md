# Metric semantics V1

`measured` means a value belongs in the denominator, including a valid `false` or zero.
`skipped` means the metric is deliberately inapplicable. `error` means an evaluator operation
failed. This distinction is part of the public result contract.

| Metric | Denominator | Invalid generated move |
| --- | --- | --- |
| `response_format_compliance` | Successful generations | `false` |
| `move_parse_success` | Successful generations | `false` |
| `legal_move_rate` | Successful generations | `false` |
| `reference_move_accuracy` | Successful generations with a reference | `false` |
| `engine_best_move_match` | Successful generations with enabled, operational engine metrics | `false` |
| `centipawn_loss` | Legal predictions with two numeric engine scores | skipped |

Provider failure skips model-quality metrics and reduces generation coverage. Missing optional
references skip reference accuracy. Mate scores remain typed signed mate distances; numeric
centipawn loss is skipped rather than using an arbitrary mate-to-centipawn conversion.

Stockfish best-move matching is exact equality with the first principal move returned by the
recorded engine configuration. It makes no equivalence claim for separately returned equal
scores. Centipawn loss compares two searches from the original position and original
side-to-move POV: the unrestricted best search and a search restricted to the predicted root
move.

