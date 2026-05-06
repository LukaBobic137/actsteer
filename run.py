import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# LOAD DATA
# -----------------------------
def load_dataset(path, ratio=1.0):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if ratio < 1.0:
        data = data[: max(1, int(len(data) * ratio))]
    return data


# -----------------------------
# MATCH (RELAXED)
# -----------------------------
def match(pred, target):
    if not isinstance(pred, str):
        return False
    pred = pred.lower().strip()
    target = str(target).lower().strip()
    return target in pred or pred in target


# -----------------------------
# FIND LAYERS ROBUSTLY
# -----------------------------
def get_layers(model):
    candidates = [
        "model.layers",
        "model.model.layers",
        "base_model.model.layers",
    ]

    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            print(f"[OK] Found layers at: {path}")
            return obj
        except Exception:
            continue

    raise RuntimeError("Could not find transformer layers!")


# -----------------------------
# STEERING VECTOR
# -----------------------------
@torch.no_grad()
def build_vector(model, tokenizer, device):
    good = "Translate English to French:"
    bad = "Random text:"

    def get_hidden(text):
        inp = tokenizer(text, return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[len(out.hidden_states)//2]
        return h[0, -1].float()

    v = get_hidden(good) - get_hidden(bad)

    print("\n[STEERING VECTOR]")
    print("Norm:", v.norm().item())

    return v


# -----------------------------
# HOOK FACTORY
# -----------------------------
def make_hook(vec):
    vec = vec.detach()

    def hook(module, input, output):
        print("[HOOK] ACTIVE")

        if isinstance(output, tuple):
            h = output[0]
            h = h + vec.to(h.device).to(h.dtype)
            return (h,) + output[1:]

        h = output + vec.to(output.device).to(output.dtype)
        return h

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
        do_sample=False
    )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------
# EVAL
# -----------------------------
def evaluate(model, tokenizer, data, device, hook_layer=None, hook=None):

    correct = 0

    if hook is not None:
        handle = hook_layer.register_forward_hook(hook)
        print("\n[INFO] Hook registered on layer")
    else:
        handle = None

    for ex in tqdm(data):
        prompt = f"Translate English to French: {ex['input']}"
        pred = generate(model, tokenizer, prompt, device)

        if match(pred, ex["output"]):
            correct += 1

    if handle:
        handle.remove()

    return correct / len(data)


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

    data = load_dataset(args.data, args.subset)

    # -------------------------
    # DETECT LAYERS
    # -------------------------
    layers = get_layers(model)
    L = len(layers)

    layer_map = {
        "l25": int(L * 0.25),
        "l50": int(L * 0.50),
        "l75": int(L * 0.75),
    }

    print("\n[MODEL DEPTH]", L)
    print("[LAYERS]", layer_map)

    # -------------------------
    # BASELINE
    # -------------------------
    print("\n=== BASELINE ===")
    baseline = evaluate(model, tokenizer, data, device)
    print("baseline:", baseline)

    # -------------------------
    # STEERING VECTOR
    # -------------------------
    vec = build_vector(model, tokenizer, device)

    results = {"baseline": baseline}

    # -------------------------
    # STEERING TESTS
    # -------------------------
    for name, idx in layer_map.items():

        print(f"\n=== {name} (layer {idx}) ===")

        layer = layers[idx]

        acc = evaluate(
            model,
            tokenizer,
            data,
            device,
            hook_layer=layer,
            hook=make_hook(vec)
        )

        print(f"{name}: {acc}")

        results[name] = acc

    # -------------------------
    # SAVE
    # -------------------------
    os.makedirs("results", exist_ok=True)

    out = "results/results_debug_steering.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\nSaved →", out)


if __name__ == "__main__":
    main()