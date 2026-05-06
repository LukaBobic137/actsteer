import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# DATA
# -----------------------------
def load_data(path, ratio=1.0):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if ratio < 1.0:
        data = data[:max(1, int(len(data) * ratio))]
    return data


# -----------------------------
# MATCH
# -----------------------------
def match(pred, target):
    if not isinstance(pred, str):
        return False
    return str(target).lower() in pred.lower()


# -----------------------------
# NORMALIZE
# -----------------------------
def normalize(v):
    return v / (v.norm() + 1e-8)


# -----------------------------
# BUILD STEERING VECTOR
# -----------------------------
@torch.no_grad()
def build_vector(model, tokenizer, device):
    pos = "Translate English to French:"
    neg = "Random unrelated text:"

    def get_hidden(text):
        inp = tokenizer(text, return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[len(out.hidden_states) // 2]
        return h[0, -1].float()

    v = get_hidden(pos) - get_hidden(neg)
    return normalize(v)


# -----------------------------
# HOOK
# -----------------------------
def make_hook(vec, alpha):
    vec = vec.detach()

    def hook(module, input, output):
        delta = alpha * vec

        if isinstance(output, tuple):
            h = output[0] + delta.to(output[0].device).to(output[0].dtype)
            return (h,) + output[1:]

        return output + delta.to(output.device).to(output.dtype)

    return hook


# -----------------------------
# GENERATION
# -----------------------------
@torch.no_grad()
def generate(model, tokenizer, prompt, device):
    inp = tokenizer(prompt, return_tensors="pt").to(device)

    out = model.generate(
        **inp,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(model, tokenizer, data, device, layer, vec, alpha):
    layer_module = model.model.layers[layer]

    hook_handle = layer_module.register_forward_hook(
        make_hook(vec, alpha)
    )

    correct = 0

    for ex in data:
        prompt = f"Translate English to French: {ex['input']}"
        pred = generate(model, tokenizer, prompt, device)

        if match(pred, ex["output"]):
            correct += 1

    hook_handle.remove()

    return correct / len(data)


# -----------------------------
# LAYER MAP
# -----------------------------
def get_layers(L):
    return {
        "l25": int(L * 0.25),
        "l50": int(L * 0.50),
        "l75": int(L * 0.75),
    }


# -----------------------------
# MAIN
# -----------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data", default="dataset_files/english-french.json")
    parser.add_argument("--subset", type=float, default=0.01)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.eval()

    data = load_data(args.data, args.subset)

    # -------------------------
    # LAYERS
    # -------------------------
    L = len(model.model.layers)
    layers = get_layers(L)

    print("\nModel layers:", L)
    print("Target layers:", layers)

    # -------------------------
    # BASELINE
    # -------------------------
    print("\n=== BASELINE ===")
    baseline_correct = 0

    for ex in data:
        prompt = f"Translate English to French: {ex['input']}"
        pred = generate(model, tokenizer, prompt, device)

        if match(pred, ex["output"]):
            baseline_correct += 1

    baseline = baseline_correct / len(data)
    print("baseline:", baseline)

    # -------------------------
    # STEERING VECTOR
    # -------------------------
    vec = build_vector(model, tokenizer, device)

    alphas = [0.05, 0.1, 0.2, 0.5]

    results = {"baseline": baseline}

    # -------------------------
    # STEERING TESTS
    # -------------------------
    for name, layer_idx in layers.items():

        best_acc = 0
        best_alpha = None

        for a in alphas:

            acc = evaluate(
                model,
                tokenizer,
                data,
                device,
                layer_idx,
                vec,
                a
            )

            if acc > best_acc:
                best_acc = acc
                best_alpha = a

        results[name] = best_acc

        print(f"{name}: {best_acc:.4f} (alpha={best_alpha})")

    # -------------------------
    # SAVE
    # -------------------------
    out_path = "results_fv.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\nSaved →", out_path)


if __name__ == "__main__":
    main()