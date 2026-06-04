import re


# === Answer normalization ===
def normalize_answer(s: str) -> str:
    """Normalize a numeric answer string for comparison."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.replace(" ", "")  # "1 000" → "1000"
    s = s.lstrip("+")  # "+42" → "42"
    try:
        num = float(s)
        s = str(int(num)) if num == int(num) else str(num)
    except ValueError:
        pass
    return s


# === Lenient substring matching (the "reward" matcher, matches Exp 1) ===
def answer_in_text(gt: str, text: str) -> bool:
    """Lenient substring match — does GT number appear anywhere in text?"""
    gt = normalize_answer(gt)
    if not gt or not text:
        return False
    text_normalized = text.replace(",", "").replace("$", "").replace("%", "")
    pattern = r"(?<![\d.-])" + re.escape(gt) + r"(?!\d|\.[\d])"
    return re.search(pattern, text_normalized) is not None


# === Strict accuracy extraction (the honest metric) ===
def extract_answer_strict(text: str) -> str | None:
    """Extract model's intended final answer. Tiered: <answer> → \\boxed → last number."""

    def last_num(s):
        nums = re.findall(r"[\d,]+(?:\.\d+)?", s)
        return nums[-1].replace(",", "") if nums else None

    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m and last_num(m.group(1)) is not None:
        return last_num(m.group(1))
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m and last_num(m.group(1)) is not None:
        return last_num(m.group(1))
    return last_num(text)


def strict_correct(gt: str, text: str) -> bool:
    """Strict correctness: does extracted answer numerically equal GT?"""
    pred = extract_answer_strict(text)
    if pred is None:
        return False
    try:
        return float(pred) == float(normalize_answer(gt))
    except ValueError:
        return False


def strict_tag_correct(gt: str, text: str) -> bool:
    """Strictest evaluator: correct number must sit inside <answer>...</answer>."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        return False
    nums = re.findall(r"[\d,]+(?:\.\d+)?", m.group(1))
    if not nums:
        return False
    try:
        return float(nums[-1].replace(",", "")) == float(normalize_answer(gt))
    except ValueError:
        return False


# === Format compliance (observational) ===
FORMAT_PATTERN = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def has_answer_format(text: str) -> bool:
    return bool(FORMAT_PATTERN.search(text))


# === Shared compute_metrics — logs all three metrics ===
def make_compute_metrics():
    """compute_metrics held CONSTANT across all 3 runs.
    Logs the training reward (mean_reward, per-run) plus accuracy under all
    three measurement evaluators (lenient / last-number / strict-tag)."""

    def compute_metrics(eval_pred):
        rewards = eval_pred.label_ids  # (N, 1) — THIS run's training reward
        completions = eval_pred.predictions
        answers = eval_pred.inputs  # ground truths (from the fork patch)
        n = len(completions)

        mean_reward = rewards[:, 0].mean().item()

        def acc(fn):
            if not n:
                return 0.0
            return sum(1 for c, gt in zip(completions, answers, strict=False) if fn(gt, c[0]["content"])) / n

        lenient_acc = acc(answer_in_text)  # substring — most gameable
        last_number_acc = acc(strict_correct)  # tiered extractor (your old "strict_accuracy")
        strict_tag_acc = acc(strict_tag_correct)  # tag-only — strictest

        format_count = sum(1 for c in completions if has_answer_format(c[0]["content"]))
        format_compliance = format_count / n if n else 0.0

        print("\n=== compute_metrics called ===")
        print(f"completions received: {n}")
        for i in range(min(3, n)):
            print(f"  [{i}] expected={answers[i]} | text={completions[i][0]['content'][:80]}")
        print(f"mean_reward (this run's reward): {mean_reward:.3f}")
        print(f"lenient_acc:     {lenient_acc:.3f}")
        print(f"last_number_acc: {last_number_acc:.3f}")
        print(f"strict_tag_acc:  {strict_tag_acc:.3f}")
        print(f"format_compliance: {format_compliance:.3f}")

        return {
            "mean_reward": mean_reward,  # per-run (training signal)
            "lenient_acc": lenient_acc,  # ┐
            "last_number_acc": last_number_acc,  # ├ held-constant measurement evaluators
            "strict_tag_acc": strict_tag_acc,  # ┘
            "format_compliance": format_compliance,
        }

    return compute_metrics


# === Dataset prep ===
SYSTEM_PROMPT = (
    "Solve the math problem step by step. Wrap your final answer in <answer> tags, e.g. <answer>42</answer>."
)


def format_sample(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"].split("####")[-1].strip(),
    }
