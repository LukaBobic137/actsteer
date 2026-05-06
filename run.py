import os
import argparse
import torch
from tqdm import tqdm

from utils.fv_dataset_loader import load_fv_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# ARGPARSE
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_dir", type=str, default="dataset_files")
    parser.add_argument("--use_subset", type=bool, default=True)
    parser.add_argument("--subset_ratio", type=float, default=0.1)

    return parser.parse_args()


# -----------------------------
# SIMPLE MATCHING (REALISTIC)
# -----------------------------
def is_correct(pred, target):
    pred = pred.lower().strip()
    target = target.lower().strip()

    return target in pred or pred.split()[0] == target.split()[0]


# -----------------------------
# GENERATION
# -----------------------------
def generate(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=0.0
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------
# EVAL
# -----------------------------
def evaluate(model, tokenizer, dataset, device):
    correct = 0
    total = 0

    results = []

    for ex in tqdm(dataset):
        prompt = ex["input"]
        target = ex["target"]

        pred = generate(model, tokenizer, prompt, device)

        ok = is_correct(pred, target)

        correct += int(ok)
        total += 1

        results.append({
            "input": prompt,
            "target": target,
            "pred": pred,
            "correct": ok
        })

    acc = correct / max(total, 1)

    return acc, results


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    tasks = ["english-french", "english-german", "english-spanish"]

    print("Loading dataset...")
    dataset = load_fv_dataset(args.data_dir, tasks)

    if args.use_subset:
        dataset = dataset[: int(len(dataset) * args.subset_ratio)]

    print(f"Dataset size: {len(dataset)}")

    print("Running evaluation...")
    acc, results = evaluate(model, tokenizer, dataset, device)

    print("\n=== RESULTS ===")
    print(f"Accuracy: {acc:.4f}")

    # save
    out_path = "results_fv.json"
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()