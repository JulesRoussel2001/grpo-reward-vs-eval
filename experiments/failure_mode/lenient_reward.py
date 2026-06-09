"""
Experiment 2 — Correctness reward only.

Hypothesis: rewarding only correctness (does GT answer appear in the
completion?) is harder to game. Loss does not collapse, mean_reward
rises modestly. Format compliance emerges as a side effect even
though it's not rewarded — an asymmetric generalization finding.

Reward:    lenient substring match (does GT number appear in completion?)
Observed:  mean_reward, strict_accuracy, format_compliance (emergent)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset
from eval_utils import (
    answer_in_text,
    format_sample,
    make_compute_metrics,
)

from trl import GRPOConfig, GRPOTrainer


def make_accuracy_reward():
    """Returns a fresh accuracy_reward function with its own first-call flag."""
    first_call = [True]

    def accuracy_reward(completions, answer, **kwargs):
        """1.0 if GT answer appears anywhere in completion text, else 0.0.

        Format is completely ignored — this isolates correctness from format.
        """
        rewards = [
            1.0 if answer_in_text(gt, c[0]["content"]) else 0.0 for c, gt in zip(completions, answer, strict=False)
        ]
        if first_call[0]:
            print("\n=== First reward_func call ===")
            print(f"completions received: {len(completions)}")
            print(f"first completion text: {completions[0][0]['content'][:150]}")
            print(f"first reward score: {rewards[0]}")
            first_call[0] = False
        return rewards

    return accuracy_reward


def main():
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].select(range(200)).map(format_sample)
    eval_dataset = dataset["test"].select(range(50)).map(format_sample)

    print("=== Dataset sample ===")
    print(f"train example prompt: {train_dataset[0]['prompt']}")
    print(f"train example answer: {train_dataset[0]['answer']}")
    print(f"eval answers (first 5): {eval_dataset['answer'][:5]}")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=make_accuracy_reward(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=make_compute_metrics(),
        args=GRPOConfig(
            seed=42,
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


if __name__ == "__main__":
    main()
