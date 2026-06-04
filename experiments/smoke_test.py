"""
Smoke test — runs format_reward experiment with minimal data/steps
to verify the full pipeline (dataset, reward, compute_metrics) works.

Run from experiments/:
    python smoke_test.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_dataset
from eval_utils import format_sample, has_answer_format, make_compute_metrics
from trl import GRPOConfig, GRPOTrainer


def make_format_reward():
    first_call = [True]

    def format_reward(completions, answer, **kwargs):
        rewards = [1.0 if has_answer_format(c[0]["content"]) else 0.0 for c in completions]
        if first_call[0]:
            print("\n=== First reward_func call ===")
            print(f"completions received: {len(completions)}")
            print(f"first completion text: {completions[0][0]['content'][:150]}")
            print(f"first reward score: {rewards[0]}")
            first_call[0] = False
        return rewards

    return format_reward


def main():
    print("Loading dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].select(range(16)).map(format_sample)
    eval_dataset = dataset["test"].select(range(8)).map(format_sample)

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    print(f"Sample prompt: {train_dataset[0]['prompt']}")
    print(f"Sample answer: {train_dataset[0]['answer']}")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=make_format_reward(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=make_compute_metrics(),
        args=GRPOConfig(
            eval_on_start=True,
            num_generations=4,
            eval_strategy="steps",
            eval_steps=5,
            logging_steps=1,
            max_steps=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            max_completion_length=256,
            report_to="none",
        ),
    )

    print("\nStarting smoke test training run...")
    trainer.train()
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
