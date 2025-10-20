from transformers import pipeline, LogitsProcessor
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import re
import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent.parent))

from mamkit.data.datasets import MMUSEDFallacy, InputMode

# Prompt helpers
SYSTEM_PROMPT = (
    "You are a fallacy classifier.\n"
    "Given an input sentence, output one digit (0-5) and nothing else.\n"
    "Digits correspond to: 0 Appeal to emotion | 1 Appeal to authority | "
    "2 Ad hominem | 3 False cause | 4 Slippery slope | 5 Slogans.\n"
    "Important: no words, punctuation or line breaks beyond that single digit.\n"
    "Examples:\n"
    "Input: They're with the insurance companies.\n"
    "Output: 2\n"
    "Input: They're the red states.\n"
    "Output: 0\n"
    "Input: That's what the economists tell us.\n"
    "Output: 1\n"
    "Input: Show up and vote.\n"
    "Output: 5\n"
    "Input: We've got two and a half million more Americans out of work now than we had when Mr. Ford took office.\n"
    "Output: 3\n"
    "Input: This administration, by going into the Star Wars system, is going to add a dangerous new escalation.\n"
    "Output: 4\n"
)


def make_prompt(text: str) -> str:
    """Wrap raw user text so the model knows exactly where the input ends and
    the answer must start."""
    return f"Input: {text}\nOutput:"

# Logits processor to restrict output vocabulary to the 6 digit tokens
class AllowedDigits(LogitsProcessor):
    """Force the model to sample only the tokens whose decoded text is 0‑5."""

    def __init__(self, tokenizer):
        self.allowed_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(6)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_ids] = 0.0  # keep only allowed ids
        return scores + mask

# Data loading helper

def load_dataset(base_data_path: Path):
    """Return the test texts and gold labels from the MMUSED‑Fallacy splits."""
    splits_dir = base_data_path / "train_test_val_mmused_fallacy"
    train_df = pd.read_pickle(splits_dir / "train_data.pkl")
    val_df   = pd.read_pickle(splits_dir / "val_data.pkl")
    test_df  = pd.read_pickle(splits_dir / "test_data.pkl")

    loader = MMUSEDFallacy(
        task_name="afc",
        input_mode=InputMode.TEXT_ONLY,
        base_data_path=base_data_path,
    )

    data = loader.build_info_from_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    return data.test.inputs, data.test.labels

def main():
    base_data_path = Path(__file__).parent.parent.parent / "data"

    # Use the tokenizer separately so we can build our logits processor
    generator = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = generator.tokenizer

    logits_processor = [AllowedDigits(tokenizer)]

    texts, gold_labels = load_dataset(base_data_path)

    predictions: list[int] = []
    invalid_examples: list[dict] = []

    for idx, text in enumerate(texts):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_prompt(text)},
        ]

        outputs = generator(
            messages,
            max_new_tokens=1,      # stop after first token
            do_sample=False,       # greedy decoding
            num_beams=1,
            logits_processor=logits_processor,
        )

        # Extract a digit 0‑5 or mark as invalid (‑1)
        generated = outputs[0]["generated_text"]
        last = generated[-1].get("content", "") if isinstance(generated, list) else generated
        match = re.search(r"\b[0-5]\b", str(last))

        if match:
            pred_digit = int(match.group())
        else:
            pred_digit = -1
            invalid_examples.append({"idx": idx, "text": text, "raw": last})

        predictions.append(pred_digit)

    # Show invalid predictions
    if invalid_examples:
        print(f"\nFound {len(invalid_examples)} invalid predictions (out of {len(texts)}):")
        for ex in invalid_examples[:10]:  # cap output for readability
            print(f"[{ex['idx']}] → {ex['raw']!r}\n   Sentence: {ex['text'][:120]}…")

    # Prepare gold and predicted arrays for metrics
    pred = np.asarray(predictions, dtype=int)
    gold_arr = np.asarray(gold_labels)

    # Handle possible NaNs in gold labels
    gold = np.where(np.isnan(gold_arr), -1, gold_arr).astype(int) if np.issubdtype(gold_arr.dtype, np.floating) else gold_arr.astype(int)

    # Display distribution
    print("\nPrediction distribution:")
    unique, counts = np.unique(pred, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")

    # Confusion matrix (excluding invalid = -1)
    valid_mask = pred != -1
    if valid_mask.any():
        cm = confusion_matrix(gold[valid_mask], pred[valid_mask], labels=list(range(6)))
        print("\nConfusion matrix (rows=gold, cols=pred):\n", cm)

    # Metrics
    label_set = sorted(set(gold) | set(pred))

    print("\nClassification report:")
    print(classification_report(gold, pred, labels=label_set, zero_division=0))

if __name__ == "__main__":
    main()
