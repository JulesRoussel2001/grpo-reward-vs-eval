grpo-reward-vs-eval
Reinforcement learning from verifiable rewards (RLVR) quietly fuses three things practitioners treat as one: the reward a model is trained against, the accuracy it is measured by, and the extractor that turns a free-form completion into the answer feeding both. This repo pulls them apart on GSM8K with GRPO and a 0.5B model, and shows the fusion hides two distinct failure modes.
Headline: a pure-format reward drives format_compliance and mean_reward to 1.0 while honest accuracy collapses from 0.228 to 0.025 — the model gets monotonically better at the proxy while getting worse at the task.
The point isn't "format rewards are bad." It's that a rising reward curve is not evidence of a more capable model, and you cannot tell the difference without a separate, audited measurement instrument. This project builds that instrument and uses it.
Why this matters
In most public GRPO/RLVR pipelines the reward function and the eval metric are the same regex or substring check. When they coincide, "reward went up" and "the model improved" become indistinguishable by construction. The three steps below separate them, quantify how trustworthy each extractor actually is, and then ask which reward trains the best model when judged by the most trustworthy one.
The three steps and what they found
Step 1 — Failure mode (experiments/failure_mode/)
Train GRPO three times — format-only, lenient-substring, and a 0.3·format + 0.7·correctness combination — under one fixed evaluation harness. The format-only reward is the clean reward-hacking case (values are start → end over steps 0→100):
rewardmean_rewardformat_compliancehonest accuracyformat-only0.438 → 1.0000.438 → 1.0000.228 → 0.025lenient0.377 → 0.4980.390 → 0.0620.258 → 0.335combined0.381 → 0.5700.390 → 0.8380.258 → 0.325
Honest accuracy is the tiered last-number extractor (logged as strict_accuracy in Step 1 — the same function Step 3 calls last_number_acc). Under the format-only reward, the model learns to emit well-formed <answer> tags containing wrong numbers.
Step 2 — Extractor faithfulness (experiments/audit/)
Calling one extractor "honest" needs evidence. A Claude judge (Anthropic tool-use for structured verdicts) identifies each completion's intended final answer; the judge is validated at 96% agreement (48/50) against hand-labels, then 500 completions are audited. Each extractor is scored against the judge:
extractorprecisionrecallF1lenient (substring)0.7030.9640.813last_number (tiered)0.9620.9140.938strict_tag (tag-only)0.9570.3140.473
The lenient matcher over-credits (57 false positives — it fires on any right-looking number in the reasoning). The strict-tag matcher under-counts (96 false negatives — it ignores correct answers written outside <answer> tags). The tiered last_number extractor is the only defensibly honest metric (F1 0.938) — which retroactively justifies reporting it in Step 1.
Step 3 — Evaluator-strictness curve (experiments/strictness_curve/)
Train GRPO with each audited extractor as the reward, then re-score every final model with the independent Step-2 judge (which was never a training target) to break the train/eval circularity. The ranking of "which reward trained the best model" is evaluator-dependent:
trained onjudge accuracystrict_tag0.480lenient0.460last_number0.320
The lenient-trained run wins by its own (last_number) extractor but the strict-tag run wins under the independent judge — the reward signal and the true objective diverge. Across every reward, the matching proxy climbs while honest accuracy stays ~0.27–0.34: at this scale the model optimizes whatever proxy it is given without becoming more correct.
Repository structure
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
The three extractors are defined once, in eval_utils.py, and imported everywhere — so "the function used to reward" and "the function used to measure" are literally the same object. make_compute_metrics is held constant across all six training runs.
Reproduce
Requires a GPU for all training and for generate_audit_sample.py; the judge and audit steps need only CPU + an Anthropic API key.
bashpip install -r requirements.txt           # pulls TRL from the fork (exposes eval_pred.inputs)
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
The fork is required because stock TRL's compute_metrics does not receive ground-truth answers; the forked branch adds the answers buffer that exposes them as eval_pred.inputs, which the held-constant harness needs.
Limitations
Single seed, single model (Qwen2.5-0.5B-Instruct), GSM8K only, 200 train / 50 eval problems, 100 steps. Step 3's effects are small relative to step-to-step validation-loss noise — the ranking is directional, not conclusive. A separate robustness sweep (in results/, logged with an earlier proxy-only harness) shows max_completion_length (128→512) is the decisive setup factor — short completions truncate before concluding — while num_generations and model scale move the proxy less; that sweep speaks only to the proxy, not to honest accuracy. The claim "the model optimizes the proxy, not the skill" is demonstrated at this scale; whether it persists at larger scale is untested here. "Reasoning capacity is the bottleneck" is offered as a hypothesis, not a finding — the experiments isolate the reward-vs-skill decoupling, not its underlying cause.
License
Apache-2.0. Built on Hugging Face TRL (Apache-2.0).