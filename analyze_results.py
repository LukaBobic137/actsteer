import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from fv_dataset_loader import load_fv_dataset


# ----------------------------
# ARGUMENTS
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_dir", type=str, default="dataset_files")

    parser.add_argument("--use_subset", action="store_true")
    parser.add_argument("--subset_ratio", type=float, default=0.05)

    parser.add_argument("--output_dir", type=str, default="results")

    return parser.parse_args()


# ----------------------------
# SIMPLE GENERATION
# ----------------------------
@torch.no_grad()
def generate(model, tokenizer, text, max_new_tokens=20):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # only continuation
    return decoded[len(text):].strip()


# ----------------------------
# SIMPLE MATCH (FV style)
# ----------------------------
def is_correct(pred, target):
    if pred is None:
        return False
    return pred.strip().split()[0].lower() == str(target).strip().split()[0].lower()


# ----------------------------
# STEERING PLACEHOLDER
# (ako imaš pravi FV hook, ovdje ga spajaš)
# ----------------------------
def apply_steering(model, layer_id):
    """
    PLACEHOLDER:
    Ako imaš FV hooking, ovo se ovdje spaja.
    Trenutno samo vraća model.
    """
    return model


# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    print("Loading dataset...")
    df = load_fv_dataset(args.data_dir)

    if args.use_subset:
        df = df.sample(frac=args.subset_ratio, random_state=42)

    print(f"Dataset size: {len(df)}")

    results = []

    # steering layers (iz paper setupa)
    steering_layers = {
        "baseline": None,
        "l25": 8,
        "l50": 16,
        "l75": 24
    }

    for row in tqdm(df.to_dict(orient="records")):

        input_text = row["input"]
        target = row["target"]

        sample = {
            "input": input_text,
            "target": target
        }

        # -------------------------
        # BASELINE
        # -------------------------
        baseline_pred = generate(model, tokenizer, input_text)
        sample["baseline"] = baseline_pred

        # -------------------------
        # STEERING RUNS
        # -------------------------
        for name, layer in steering_layers.items():

            if name == "baseline":
                continue

            steered_model = apply_steering(model, layer)

            pred = generate(steered_model, tokenizer, input_text)

            sample[f"{name}_output"] = pred

        results.append(sample)

    # ----------------------------
    # SAVE
    # ----------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, "results_fv.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()