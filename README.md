# grpo-reward-vs-eval

A small, reproducible testbed that **separates and audits** the three things RLVR pipelines usually fuse into one regex: the **reward** a model is trained against, the **metric** it is measured by, and the **extractor** that turns a free-form completion into the answer feeding both.

When those three coincide — as they do in most public GRPO/RLVR pipelines — "reward went up" and "the model improved" become indistinguishable *by construction*. This repo pulls them apart on GSM8K (Qwen2.5-0.5B-Instruct, GRPO via TRL) and uses an independently-audited measurement instrument to tell capability gains apart from proxy-gaming.

## What this is, and what it is not

**This is** an honest, end-to-end demonstration and a reusable method:
1. a harness in which the *same* extractor functions are used in both roles (reward and metric), so the fusion is explicit and controllable;
2. an **audited** extractor — its faithfulness is measured against a validated judge before any number it produces is trusted;
3. a cheap, pass@k-free **diagnostic** for distinguishing "the model got more correct" from "the extractor got better at finding answers."

**This is not** a claim of new phenomena. The failure modes shown here — reward hacking, RLVR sharpening existing competence rather than installing new competence, and answer-extractor choice swinging measured accuracy — are all **established results** (see [Relation to prior work](#relation-to-prior-work)). The contribution is methodological and pedagogical: a clean, controlled, reproducible place to *see* and *measure* these effects, with the verifier itself audited rather than assumed.

## The two things worth your attention

If you read nothing else:

- **Decoupled, audited extractors in one fixed harness.** Three extractors are defined once in `eval_utils.py` and imported everywhere, so "the function used to reward" and "the function used to measure" are literally the same Python object. Each extractor's faithfulness is then quantified (precision/recall/F1) against a Claude judge that is *itself validated* against hand-labels (48/50 = 96% agreement) before use. Most pipelines pick an extractor and trust it; this one measures whether the extractor is right first.

- **The recall-convergence diagnostic (preliminary).** Run two extractors of *different recall* side-by-side during training. If the proxy-matched extractor's accuracy climbs only *up to* the value a recall-robust extractor was already reporting — and then stops — the gain was the extractor's recall catching up to a fixed pool of already-correct answers, **not** new correctness. This reaches the same conclusion as the pass@k "RLVR elicits, doesn't instill" line of work (Yue et al., 2025) *without* needing large-k sampling or a base-model comparison. **Status: a single-seed, n=50 signal — suggestive, not confirmed.** See [Step 3](#step-3--extractor-as-reward-experimentsstrictness_curve).

## A note on naming (read this before the tables)

The project evolved as it ran, and one log key is misleading: in Step 1 the honest metric is logged as **`strict_accuracy`**, but it is the **tiered last-number extractor** — the *same* function Step 3 calls **`last_number_acc`**. It is **not** the tag-only extractor that Steps 2–3 call **`strict_tag`**. Concretely: a tag-only extractor scores ~0 on the tagless step-0 completions, so the 0.228 reported at step 0 of Step 1 can only be the last-number extractor.

To avoid confusion, throughout this README the honest metric is written **`last_number`** (a.k.a. honest accuracy) regardless of how the raw logs label it. *If you fork this, rename the log key `strict_accuracy` → `last_number_acc`.*

| name used here | what it does | role(s) |
| --- | --- | --- |
| `lenient` | substring: does the GT number appear anywhere in the completion? | reward + metric |
| `last_number` | tiered last-number extraction (the honest metric) | reward + metric |
| `strict_tag` | only the number inside `<answer>…</answer>` | reward + metric |

---

## Step 1 — Failure mode (`experiments/failure_mode/`)

Train GRPO three times — **format-only**, **lenient-substring**, and a **0.3·format + 0.7·correctness** combination — under one fixed evaluation harness. Values are start → end over steps 0→100.

| reward | mean_reward | format_compliance | honest accuracy (`last_number`) |
| --- | --- | --- | --- |
| format-only | 0.438 → 1.000 | 0.438 → 1.000 | **0.228 → 0.025** |
| lenient | 0.377 → 0.498 | 0.390 → 0.062 | 0.258 → 0.335 |
| combined | 0.381 → 0.570 | 0.390 → 0.838 | 0.258 → 0.325 |

The format-only run is the clean reward-hacking case: the proxy saturates to 1.0 while honest accuracy **collapses below its starting point**. This is worth stating precisely, because it is more than "accuracy failed to improve." At step 0, with free-form reasoning, the model solves ~23% of problems. The format-only reward removes any incentive to produce the chain-of-thought that was *load-bearing for correctness*, so the model learns to emit a bare `<answer>NUMBER</answer>` with the reasoning stripped out — and accuracy craters to 2.5%. The reward **destroyed competence the base model already had.** (This is a textbook reward-hacking / capability-narrowing outcome; see prior work — it is shown here as a clean controlled instance, not as a new effect.)

Two mechanistic notes, both expected GRPO behavior rather than findings:
- **Why the collapse is permanent.** Once the format reward saturates, every completion in a group earns identical reward → the group-relative advantage is zero → the gradient vanishes. The training-loss column reads a literal `0.000000` from step 20 on. The model cannot recover even with more steps.
- **Why combined keeps learning.** The combined reward never saturates (caps ~0.57, loss stays nonzero): the format term gives a dense early gradient while the correctness term keeps the reward off the ceiling, preserving a usable advantage signal. This is the standard motivation for DeepSeek-style format+accuracy rewards.

Also observe the **mirror-image surface dynamics**: format-only drives compliance to 1.0; lenient *abandons* the tag entirely (compliance 0.39 → 0.06, since tags earn nothing); combined preserves both. The model is highly plastic on *form* and rigid on *skill* — it reshapes whatever surface the reward pays for, while honest accuracy stays pinned near the base rate.

## Step 2 — Extractor faithfulness (`experiments/audit/`)

Calling one extractor "honest" requires evidence. A Claude judge (Anthropic tool-use for structured verdicts) identifies each completion's *intended* final answer. The judge is **validated at 96% agreement (48/50)** against hand-labels, then used to audit 500 completions. Each extractor is scored against the judge:

| extractor | precision | recall | F1 | TP / FP / FN / TN |
| --- | --- | --- | --- | --- |
| `lenient` | 0.703 | 0.964 | 0.813 | 135 / 57 / 5 / 303 |
| `last_number` | **0.962** | 0.914 | **0.938** | 128 / 5 / 12 / 355 |
| `strict_tag` | 0.957 | 0.314 | 0.473 | 44 / 2 / 96 / 358 |

- `lenient` **over-credits** (57 false positives): it fires on any right-looking number in the scratch work, even when the model never commits to it.
- `strict_tag` **under-counts** (96 false negatives): it ignores correct answers written outside `<answer>` tags.
- `last_number` is the only **balanced, defensibly honest** ruler (F1 0.938, 5 FP / 12 FN). This is what justifies treating it as the honest metric in Steps 1 and 3.

The two judge disagreements are logged and inspected; both are genuinely ambiguous completions (one truncated/incoherent, one with omitted cost components), not judge errors.

## Step 3 — Extractor-as-reward (`experiments/strictness_curve/`)

Train GRPO with each audited extractor *as the reward*, then re-score every final model with the **independent Step-2 judge** (never a training target) to break train/eval circularity.

### What is robust

**The most faithful extractor for *measuring* is the worst extractor for *rewarding*, and three of four instruments agree.** On the held-out set, the **`last_number`-trained** model is ranked worst by three independent instruments (lenient ruler, last_number ruler, and the judge):

| trained on | `lenient` ruler | `last_number` ruler | `strict_tag` ruler | **judge** |
| --- | --- | --- | --- | --- |
| `lenient` | 0.58 | 0.50 | 0.00 | 0.460 |
| `last_number` | 0.44 | 0.32 | 0.02 | **0.320** |
| `strict_tag` | 0.56 | 0.46 | 0.46 | 0.480 |

Three of the four instruments rank the `last_number`-trained model worst (lenient ruler, last_number ruler, and the independent judge — margin 0.12–0.14 to the next-worst). The lone exception is the `strict_tag` ruler, where both the `last_number`- and `lenient`-trained models score near zero (0.02 and 0.00) — and that ruler is precisely the one the Step-2 audit flagged as least faithful (F1 0.473), so the exception is expected rather than contradictory. Because this agreement is *measurement-method-independent*, it is among the firmer results in the repo — and a direct, controlled instance of the published principle that a verifier's static quality does not predict its value as an RL reward (the objective-mismatch line of work; "From Accuracy to Robustness").

**Self-honesty differs sharply by reward.** The `lenient`-trained model over-credits *itself*: 0.58 by the lenient ruler vs **0.46 by the judge** — a 12-point proxy/truth gap. The `strict_tag`-trained model scores 0.46 by its own ruler and 0.48 by the judge — **almost no gap**. This is exactly what Step 2 predicts (lenient's false positives inflate its own reading; strict_tag's high precision does not).

### What is preliminary — do not over-read

The headline-tempting result is the *top* of the ranking: `strict_tag` (judge 0.480) edges `lenient` (judge 0.460). **That is 24/50 vs 23/50 — one completion, at n=50, inside the noise.** It is reported as **directional, not conclusive**, and should not be cited as "strict_tag is the best reward." The robust claim is about the *bottom* of the ranking (above) and the *self-honesty gap*, not this flip.

### The recall-convergence diagnostic (single-seed signal)

In the `strict_tag` run, measured on the consistent training-time eval (all metrics computed the same way at each step):

| step | format_compliance | `strict_tag_acc` | `last_number_acc` (honest) |
| --- | --- | --- | --- |
| 0 | 0.417 | 0.100 | 0.265 |
| … | 0.738 | 0.177 | 0.242 |
| … | 0.860 | 0.212 | 0.237 |
| … | 0.887 | 0.260 | 0.273 |
| final | 0.890 | 0.270 | 0.282 |

`strict_tag_acc` climbs 0.10 → 0.27 (×2.7) and `format_compliance` climbs 0.42 → 0.89 — but the recall-robust `last_number` stays flat (~0.27). `strict_tag_acc` converges **up to** the `last_number` ceiling and stops there. Interpretation: the model relocated answers it was *already getting right* into the tagged slot, raising the tag-extractor's recall; it did not become more correct. (If it were stuffing tags with random numbers, `strict_tag_acc` could not reach the GT-match ceiling.) Across all three rewards, honest accuracy stays in a narrow ~0.24–0.34 band: at this scale the model optimizes whatever proxy it is given without becoming more correct.

**Caveat:** the final held-out eval uses **greedy** decoding while training-time eval averages **8 sampled** completions per prompt, so the final per-run numbers (e.g. `strict_tag` 0.46) are not directly comparable to the training-time trajectory (0.27). The convergence pattern above is read on the internally-consistent training-time trajectory; confirming it requires multi-seed runs and matched decoding.

---

## Repository structure

```
grpo-reward-vs-eval/
├── experiments/
│   ├── eval_utils.py            shared spine: extractors + compute_metrics + dataset prep
│   ├── failure_mode/            Step 1 — reward varies, evaluator fixed
│   │   ├── format_reward.py  lenient_reward.py  combined_reward.py
│   ├── audit/                   Step 2 — extractor faithfulness pipeline
│   │   ├── generate_audit_sample.py  split_sample.py  build_labeling_csv.py
│   │   ├── check_labels.py  validate_judge.py  audit_extractors.py
│   └── strictness_curve/        Step 3 — extractor-as-reward varies
│       ├── run_strictness_curve.py  judge_final_models.py
├── results/                     compact committed metrics (mirrors experiments/)
├── requirements.txt             installs TRL from the forked branch (answers buffer)
└── .gitignore
```

The three extractors live once in `eval_utils.py`; `make_compute_metrics` is held constant across all six training runs. **Built-in sanity checks** (worth noting when reading the logs): in every Step-3 run `mean_reward` equals the matching extractor column *exactly* — proof the reward and that metric are the same object; all three Step-3 runs share identical step-0 metrics — proof the harness is deterministic, so any divergence comes from the reward, not the setup.

## Reproduce

Requires a GPU for all training and for `generate_audit_sample.py`; the judge and audit steps need only CPU + an Anthropic API key.

```bash
pip install -r requirements.txt           # pulls TRL from the fork (exposes eval_pred.inputs)
export ANTHROPIC_API_KEY=sk-ant-...        # needed for every judge call

# Step 1 — failure mode (from repo root)
python experiments/failure_mode/format_reward.py
python experiments/failure_mode/lenient_reward.py
python experiments/failure_mode/combined_reward.py

# Step 2 — extractor audit (run from experiments/audit/)
python generate_audit_sample.py --n 550 --output audit_sample_full.json
python split_sample.py --input audit_sample_full.json --n-validation 50
python build_labeling_csv.py              # writes validation_to_label.csv — hand-label my_label
python check_labels.py                    # confirm no blank labels
cp validation_to_label.csv validation_labeled.csv   # name bridge for validate_judge
python validate_judge.py --labeled validation_labeled.csv --sample judge_validation_sample.json
python audit_extractors.py --input audit_sample.json --output audit_results.json

# Step 3 — strictness curve (run from experiments/strictness_curve/)
python run_strictness_curve.py lenient
python run_strictness_curve.py last_number
python run_strictness_curve.py strict_tag
python judge_final_models.py              # independent judge on the three final models
```

The TRL fork is required because stock `compute_metrics` does not receive ground-truth answers; the forked branch adds an answers buffer that exposes them as `eval_pred.inputs`, which the held-constant harness needs. This patch is proposed upstream as [huggingface/trl#5790](https://github.com/huggingface/trl/pull/5790) (open PR, fixing issue #2959); once merged, the released TRL package can be used directly without the fork.

## Limitations

These are load-bearing, not boilerplate. Read them before drawing conclusions.

- **Statistical power.** Single seed, single model (Qwen2.5-0.5B-Instruct), GSM8K only, 200 train / 50 eval problems, 100 steps. The Step-3 top-of-ranking flip (0.480 vs 0.460) is **one completion** and is reported as directional only. The recall-convergence diagnostic is a **single-seed** signal. Both need multi-seed runs at larger n before they are results rather than observations.
- **Decoding mismatch.** Final per-run eval is greedy; training-time eval averages 8 sampled completions. Cross-regime numbers are not directly comparable; the per-run final accuracies should be re-run under matched decoding.
- **Scale.** "The model optimizes the proxy, not the skill" is demonstrated *at this scale*, where base competence on GSM8K is low. The same separation would be expected to weaken as the base model gets stronger (more real correctness to elicit) — untested here, and a reason this 0.5B setup is *good for exposing the effect* but not for claiming it scales.
- **An earlier hypothesis this data overturned.** An initial guess was that a strict reward would give the optimizer no gradient and collapse like the format-only run. It did not — the `strict_tag` reward trained stably (compliance 0.42 → 0.89). That guess is not asserted anywhere as a finding; it is noted here as corrected by the experiment.
- **Cause vs. effect.** "Reasoning capacity is the bottleneck" is a hypothesis, not a finding. These experiments isolate the reward-vs-skill *decoupling*; they do not establish its underlying cause.
- **A separate proxy-only sweep** (in `results/`, logged with an earlier proxy-only harness) shows `max_completion_length` (128→512) is the decisive setup factor — short completions truncate before concluding — while `num_generations` and model scale move the proxy less. That sweep speaks only to the proxy, not to honest accuracy.

## Relation to prior work

This repo *replicates and packages* known results; it does not claim them as new. The honest framing is "a controlled, audited testbed for effects the field has already established," plus a cheap diagnostic.

- **RLVR elicits, rather than instills, capability.** Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* (arXiv:2504.13837). The pass@k result that this repo's recall-convergence diagnostic reaches by a cheaper route.
- **A good evaluator is not necessarily a good reward.** Lambert & Calandra, *The Alignment Ceiling: Objective Mismatch in RLHF* (arXiv:2311.00168) — which also prescribes holding the evaluation regime fixed while varying the reward, the exact design used here. Also *From Accuracy to Robustness: ... Rule- and Model-based Verifiers in Mathematical Reasoning* (arXiv:2505.22203).
- **Answer-extractor choice swings measured accuracy.** *xFinder* (arXiv:2405.11874) reports 40+ point GSM8K swings from extractor choice alone; *Let Me Speak Freely?* (arXiv:2408.02442) validates an LLM parser against human labels (~98%). The Step-1/Step-3 cross-ruler swings (e.g. one model scoring 0.50 by `last_number` and 0.00 by `strict_tag`) are smaller-scale instances.
- **GRPO and verifiable rewards.** Shao et al., *DeepSeekMath* (arXiv:2402.03300), introducing GRPO and the group-relative advantage whose collapse-on-saturation is visible in Step 1.

## License

Apache-2.0. Built on Hugging Face TRL (Apache-2.0).
