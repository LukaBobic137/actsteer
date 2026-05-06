import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# DATA LOADER
# ----------------------------
def load_dataset(path, subset_ratio=1.0):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if subset_ratio < 1.0:
        n = max(1, int(len(data) * subset_ratio))
        data = data[:n]

    return data


# ----------------------------
# STEERING VECTOR (simple version)
# ----------------------------
@torch.no_grad()
def build_dummy_steering_vector(model, tokenizer, device):
    """
    NOTE:
    This is a placeholder.
    Real version = mean(good activations - bad activations)

    For now we use a small learned proxy:
    direction from random prompts.
    """

    text_good = "translate English to French:"
    text_bad = "random text generation task:"

    def get_activation(text):
        inputs = tokenizer(text, return_tensors="pt").to(device)

        outs = model(**inputs, output_hidden_states=True)
        # take middle layer
        h = outs.hidden_states[len(outs.hidden_states) // 2]
        return h[0, -1, :].float()

    v = get_activation(text_good) - get_activation(text_bad)
    return v


# ----------------------------
# HOOK FACTORY
# ----------------------------
def get_hook(steering_vector, alpha=0.8):
    steering_vector = steering_vector.detach()

    def hook(module, input, output):
        # output: (batch, seq, hidden)
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + alpha * steering_vector.to(hidden.device).to(hidden.dtype)
            return (hidden,) + output[1:]
        else:
            return output + alpha * steering_vector.to(output.device).to(output.dtype)

    return hook


# ----------------------------
# GENERATION
# ----------------------------
@torch.no_grad()
def generate(model, tokenizer, text, device, max_new_tokens=20):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# ----------------------------
# MATCHING (simple eval)
# ----------------------------
def match(pred, target):
    if not isinstance(pred, str):
        return False
    return pred.strip().lower().split()[0] == target.strip().lower().split()[0]


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data", default="dataset_files/english-french.json")
    parser.add_argument("--subset_ratio", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    model.eval()

    # load data
    data = load_dataset(args.data, args.subset_ratio)

    # build steering vector
    steering_vector = build_dummy_steering_vector(model, tokenizer, device)

    # choose layer (middle layer heuristic)
    layer_id = len(model.model.layers) // 2
    target_layer = model.model.layers[layer_id]

    # register hook
    hook = target_layer.register_forward_hook(
        get_hook(steering_vector, args.alpha)
    )

    results = []
    correct = 0

    for ex in tqdm(data):
        inp = ex["input"]
        tgt = ex["output"] if "output" in ex else ex["target"]

        prompt = f"Translate English to French: {inp}"

        pred = generate(model, tokenizer, prompt, device)

        ok = match(pred, tgt)
        correct += int(ok)

        results.append({
            "input": inp,
            "target": tgt,
            "pred": pred,
            "correct": ok
        })

    hook.remove()

    acc = correct / len(data)

    print("\n=== RESULTS ===")
    print(f"Accuracy: {acc:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/results_fv.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()