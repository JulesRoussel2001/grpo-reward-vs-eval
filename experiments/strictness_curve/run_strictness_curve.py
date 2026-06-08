"""
Step 3 (strictness_curve) — train GRPO three times, once per extractor-as-reward.
 
Runs the evaluator-strictness curve: the SAME three extractors that Step 2
audited are used here as the *reward signal* (lenient / last_number / strict_tag).
The evaluation harness (make_compute_metrics) is held constant and identical to
Step 1's, so any difference is attributable to the reward, not the measurement.
 
This file owns ONLY what is unique to Step 3:
  - the three reward factories (which wrap the canonical extractors)
  - the dataset load
  - run_experiment(): train, save the model, regenerate greedy eval completions
    to runs/<name>/eval_completions.json  (the input judge_final_models.py needs)
 
Everything else (the extractors, compute_metrics, dataset prep) is imported from
experiments/eval_utils.py — one definition, shared with Steps 1 and 2.
 
Usage:
    python run_strictness_curve.py            # all three
    python run_strictness_curve.py lenient    # just one
"""
 
import gc
import json
import os
import sys
 
# --- path shim: reach experiments/eval_utils.py (one level up) ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# -----------------------------------------------------------------------------
 
import torch
from datasets import load_dataset
from eval_utils import (
    answer_in_text,  # lenient matcher
    format_sample,
    make_compute_metrics,
    strict_correct as last_number_correct,  # tiered last-number extractor
    strict_tag_correct,  # tag-only matcher
)
 
from trl import GRPOConfig, GRPOTrainer
 
 
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
 
 
# --- the three reward factories (the only Step-3-specific logic) -------------
def make_lenient_reward():
    first_call = [True]
 
    def lenient_reward(completions, answer, **kwargs):
        rewards = [
            1.0 if answer_in_text(gt, c[0]["content"]) else 0.0 for c, gt in zip(completions, answer, strict=False)
        ]
        if first_call[0]:
            print(f"\n=== First reward call (lenient) === score[0]={rewards[0]}")
            first_call[0] = False
        return rewards
 
    return lenient_reward
 
 
def make_last_number_reward():
    first_call = [True]
 
    def last_number_reward(completions, answer, **kwargs):
        rewards = [
            1.0 if last_number_correct(gt, c[0]["content"]) else 0.0
            for c, gt in zip(completions, answer, strict=False)
        ]
        if first_call[0]:
            print(f"\n=== First reward call (last_number) === score[0]={rewards[0]}")
            first_call[0] = False
        return rewards
 
    return last_number_reward
 
 
def make_strict_tag_reward():
    first_call = [True]
 
    def strict_tag_reward(completions, answer, **kwargs):
        rewards = [
            1.0 if strict_tag_correct(gt, c[0]["content"]) else 0.0
            for c, gt in zip(completions, answer, strict=False)
        ]
        if first_call[0]:
            print(f"\n=== First reward call (strict_tag) === score[0]={rewards[0]}")
            first_call[0] = False
        return rewards
 
    return strict_tag_reward
 
 
REWARDS = {
    "lenient": make_lenient_reward,
    "last_number": make_last_number_reward,
    "strict_tag": make_strict_tag_reward,
}
 
 
# --- load data once ----------------------------------------------------------
def load_data():
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].select(range(200)).map(format_sample)
    eval_dataset = dataset["test"].select(range(50)).map(format_sample)
    print("eval answers (first 5):", eval_dataset["answer"][:5])
    return train_dataset, eval_dataset
 
 
# --- one experiment: train, save model, dump greedy eval completions ---------
def run_experiment(reward_name, train_dataset, eval_dataset):
    print(f"\n########## RUN: {reward_name} ##########")
    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=REWARDS[reward_name](),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=make_compute_metrics(),
        args=GRPOConfig(
            seed=0,
            eval_on_start=True,
            num_generations=8,
            eval_strategy="steps",
            eval_steps=20,
            logging_steps=5,
            max_steps=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            max_completion_length=512,
            report_to="none",
        ),
    )
    trainer.train()
 
    out_dir = os.path.join(RUNS_DIR, reward_name)
    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(os.path.join(out_dir, "model"))
 
    tok = trainer.processing_class
    model = trainer.model.eval()
    dev = next(model.parameters()).device
    recs = []
    for ex in eval_dataset:
        txt = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        enc = tok(txt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id
            )
        comp = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        recs.append({"problem": ex["prompt"][-1]["content"], "gt": ex["answer"], "completion": comp})
 
    with open(os.path.join(out_dir, "eval_completions.json"), "w") as f:
        json.dump(recs, f, indent=2)
    print(f"saved {len(recs)} completions -> {out_dir}/eval_completions.json")
    print("first completion preview:", recs[0]["completion"][:200])
 
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
 
def main(which=None):
    names = [which] if which else list(REWARDS)
    for nm in names:
        if nm not in REWARDS:
            raise SystemExit(f"unknown reward '{nm}'. choose from {list(REWARDS)}")
    train_dataset, eval_dataset = load_data()
    for nm in names:
        run_experiment(nm, train_dataset, eval_dataset)
 
 
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)