# results/

Compact, committed metrics mirroring `experiments/`. These JSON files are the
single source of truth for every number reported in the top-level README. Raw
training console output is **not** committed here (it is large and noisy); if you
want it for audit, keep it under `results/raw_logs/` and `.gitignore` it, or
attach it to a release.

```
results/
├── summary.json                 headline numbers for all steps (quick read)
├── failure_mode/
│   ├── format_only.json         Step 1, format-only reward
│   ├── lenient.json             Step 1, lenient-substring reward
│   └── combined.json            Step 1, 0.3*format + 0.7*correctness reward
├── audit/
│   ├── judge_validation.json    Step 2, judge vs hand-labels (48/50)
│   └── extractor_audit.json     Step 2, P/R/F1 of each extractor vs judge (n=500)
└── strictness_curve/
    ├── lenient.json             Step 3, lenient-as-reward training trajectory
    ├── last_number.json         Step 3, last_number-as-reward training trajectory
    ├── strict_tag.json          Step 3, strict_tag-as-reward training trajectory
    └── judge_final.json         Step 3, independent judge on the 3 final models
```

## Naming caveat (important)

In the Step-1 files, the honest metric appears in the **raw logs** under the key
`strict_accuracy`, but it is the **tiered last-number extractor** — the same
function Step 3 calls `last_number_acc`. It is **not** the tag-only `strict_tag`
extractor. In these JSON files that metric is stored under the unambiguous key
**`honest_accuracy`**. Each affected file repeats this in a `naming_note` field.

## Field conventions

- `eval_trajectory[*].step` — training step at which `compute_metrics` ran. The
  logs do not print the step number on each eval call; steps are **inferred** from
  the fact that there are exactly 6 eval calls aligned with the 6-row validation
  table (steps 0, 20, 40, 60, 80, 100, i.e. eval every 20 steps). Flagged per file
  via `step_inference`.
- `validation_loss[*].training_loss = null` corresponds to the raw "No log" entry
  at step 0.
- All metric values are transcribed verbatim from the logs (no rounding applied
  beyond what the logs already show).

## Known measurement caveat (Step 3)

The per-run *final* accuracies in `judge_final.json` are measured under **greedy**
decoding, while the `eval_trajectory` values inside each strictness_curve run are
the mean over **8 sampled** completions per prompt. The two regimes are not
directly comparable (e.g. strict_tag's training-time `strict_tag_acc` ~0.27 vs its
final greedy `strict_tag` 0.46). See the top-level README for why this matters.

## Provenance

- Model: Qwen2.5-0.5B-Instruct · Dataset: GSM8K · 200 train / 50 eval problems
- 100 GRPO steps · 8 generations per prompt · Step 1 (failure_mode) seed 42,
  Step 3 (strictness_curve) seed 0. The two steps are never compared against each
  other at a fixed seed, so this difference does not affect any reported result.
- Step 1 and Step 3 use different eval problem sets (Step 1: the "pie" item,
  expected=26; Step 3: the "candle" item, expected=20), so absolute base rates are
  not identical across the two steps.
