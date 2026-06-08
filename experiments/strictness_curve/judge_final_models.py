"""
Step 3 (strictness_curve) — judge the final models, breaking train/eval circularity.
 
Each Step-3 run was trained on one extractor as its reward, so scoring that run
with the same extractor flatters it by construction. This script re-scores every
run's saved completions with the INDEPENDENT, Step-2-validated Claude judge, which
was never a training target.
 
It re-implements NOTHING:
  - extractors come from experiments/eval_utils.py     (one definition)
  - judge_completion comes from experiments/audit/audit_extractors.py
    (the EXACT judge that validate_judge.py validated at ~96% agreement — so the
     "validated judge" label still applies here)
 
PREREQUISITE: run_strictness_curve.py has produced runs/<name>/eval_completions.json
for each reward (records shaped {"problem","gt","completion"}).
 
Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python judge_final_models.py
"""
 
import json
import os
import sys
 
# --- path shims: reach eval_utils (one up) and audit/ (one up, then audit) ---
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))  # experiments/      -> eval_utils
sys.path.insert(0, os.path.join(_HERE, "..", "audit"))  # experiments/audit -> audit_extractors
# -----------------------------------------------------------------------------
 
from anthropic import Anthropic
 
from audit_extractors import judge_completion  # the validated Step-2 judge, unchanged
from eval_utils import (
    answer_in_text,
    strict_correct,
    strict_tag_correct,
)
 
 
EXTRACTORS = {
    "lenient": answer_in_text,
    "last_number": strict_correct,
    "strict_tag": strict_tag_correct,
}
 
RUNS_DIR = os.path.join(_HERE, "runs")
RUNS = {
    "lenient": os.path.join(RUNS_DIR, "lenient", "eval_completions.json"),
    "last_number": os.path.join(RUNS_DIR, "last_number", "eval_completions.json"),
    "strict_tag": os.path.join(RUNS_DIR, "strict_tag", "eval_completions.json"),
}
 
JUDGE_MODEL = "claude-haiku-4-5-20251001"
 
 
def main():
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    summary = {}
 
    for label, path in RUNS.items():
        with open(path) as f:
            records = json.load(f)
        n = len(records)
        print(f"\n=== Run: {label} ({n} completions) ===")
 
        ext = {name: sum(fn(r["gt"], r["completion"]) for r in records) / n for name, fn in EXTRACTORS.items()}
        print(f"  extractor accs: {{ {', '.join(f'{k}: {v:.3f}' for k, v in ext.items())} }}")
 
        judgments, correct = [], 0
        for i, r in enumerate(records):
            j = judge_completion(client, r["problem"], r["gt"], r["completion"], model=JUDGE_MODEL)
            judgments.append({**r, **j})
            correct += 1 if j.get("is_correct") else 0
            if (i + 1) % 25 == 0:
                print(f"  judged {i + 1}/{n}")
 
        judge_acc = correct / n if n else 0.0
        print(f"  judge acc:      {judge_acc:.3f}")
        summary[label] = {"extractor": ext, "judge": judge_acc, "n": n}
 
        with open(os.path.join(RUNS_DIR, f"judge_detail_{label}.json"), "w") as f:
            json.dump(judgments, f, indent=2)
 
    print("\n=== Summary: independent judge accuracy per training reward ===")
    for label, s in sorted(summary.items(), key=lambda kv: -kv[1]["judge"]):
        e = s["extractor"]
        print(
            f"  {label:>12}: judge={s['judge']:.3f}  "
            f"(lenient={e['lenient']:.3f}, last_number={e['last_number']:.3f}, strict_tag={e['strict_tag']:.3f})"
        )
 
    with open(os.path.join(RUNS_DIR, "judge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
 
 
if __name__ == "__main__":
    main()