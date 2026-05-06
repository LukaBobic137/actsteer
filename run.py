import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset(path, subset_ratio=1.0):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if subset_ratio < 1.0:
        n = max(1, int(len(data) * subset_ratio))
        data = data[:n]

    return data


# -----------------------------
# MATCH (RELAXED = IMPORTANT)
# -----------------------------
def match(pred, target):
    if not isinstance(pred, str):
        return False

    pred = pred.lower().strip()
    target = str(target).lower().strip()

    # fuzzy containment is key for translation tasks
    return target in pred or pred in target


# -----------------------------
# STEERING VECTOR (SIMPLE PROXY)
# -----------------------------
@torch.no_grad()
def build_steering_vector(model, tokenizer, device):
    good_prompt = "Translate English to French:"
    bad_prompt = "Random continuation:"

    def get_hidden(text):
        inp = tokenizer(text, return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[len(out.hidden_states)//2]
        return h[0, -1, :].float()

    return get_hidden(good_prompt) - get_hidden(bad_prompt)


# -----------------------------
# HOOK
# -----------------------------
def make_hook(vec):
    vec = vec.detach()

    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + vec.to(h.device).to(h.dtype)
            return (h,) + output[1:]
        else:
            return output + vec.to(output.device).to(output.dtype)

    return hook


# -----------------------------
# GENERATION
# -----------------------------
@torch.no_grad()
def generate(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

    # decode FULL output (simple + robust)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data", default="dataset_files/english-french.json")
    parser.add_argument("--subset_ratio", type=float, default=0.01)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()

    # dataset
    data = load_dataset(args.data, args.subset_ratio)

    # steering vector
    steering_vec = build_steering_vector(model, tokenizer, device)

    L = len(model.model.layers)

    layer_map = {
        "l25": int(L * 0.25),
        "l50": int(L * 0.50),
        "l75": int(L * 0.75),
    }

    results = {}

    # -----------------------------
    # BASELINE
    # -----------------------------
    print("\nRunning baseline...")

    baseline_correct = 0

    for ex in tqdm(data):
        prompt = f"Translate English to French: {ex['input']}"
        pred = generate(model, tokenizer, prompt, device)

        if match(pred, ex.get("output", ex.get("target"))):
            baseline_correct += 1

    results["baseline"] = baseline_correct / len(data)

    # -----------------------------
    # STEERING RUNS
    # -----------------------------
    for name, layer_idx in layer_map.items():

        print(f"\nRunning {name} at layer {layer_idx}...")

        hook_handle = model.model.layers[layer_idx].register_forward_hook(
            make_hook(steering_vec)
        )

        correct = 0

        for ex in tqdm(data):
            prompt = f"Translate English to French: {ex['input']}"
            pred = generate(model, tokenizer, prompt, device)

            if match(pred, ex.get("output", ex.get("target"))):
                correct += 1

        hook_handle.remove()

        results[name] = correct / len(data)

    # -----------------------------
    # SAVE
    # -----------------------------
    os.makedirs("results", exist_ok=True)

    out_path = "results/results_steering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()