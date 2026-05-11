import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# LOAD JSON
# -----------------------------
def load_json(path, ratio=1.0):
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
# CORRUPT (BAD EXAMPLES)
# -----------------------------
def corrupt(text):
    words = text.split()
    if len(words) > 1:
        words = words[::-1]  # reverse
    return " ".join(words)


# -----------------------------
# GET HIDDEN STATE
# -----------------------------
@torch.no_grad()
def get_hidden(model, tokenizer, text, device):
    inp = tokenizer(text, return_tensors="pt").to(device)
    out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[len(out.hidden_states)//2]
    return h[0, -1].float()


# -----------------------------
# CONTRASTIVE VECTOR
# -----------------------------
@torch.no_grad()
def build_vector(model, tokenizer, data, device):

    good_vecs = []
    bad_vecs = []

    for ex in data:

        good_text = f"Translate English to French: {ex['input']}"
        bad_text = corrupt(good_text)

        good_vecs.append(get_hidden(model, tokenizer, good_text, device))
        bad_vecs.append(get_hidden(model, tokenizer, bad_text, device))

    mu_good = torch.stack(good_vecs).mean(0)
    mu_bad = torch.stack(bad_vecs).mean(0)

    v = mu_good - mu_bad
    return v / (v.norm() + 1e-8)


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
# GENERATE
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
# EVAL
# -----------------------------
def evaluate(model, tokenizer, data, device, layer, vec, alpha, prompt_prefix):

    hook = model.model.layers[layer].register_forward_hook(
        make_hook(vec, alpha)
    )

    correct = 0

    for ex in data:
        prompt = f"{prompt_prefix}: {ex['input']}"
        pred = generate(model, tokenizer, prompt, device)

        if match(pred, ex["output"]):
            correct += 1

    hook.remove()

    return correct / len(data)


# -----------------------------
# LAYERS
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
def run_task(model, tokenizer, device, data, name, vec, layers):

    print(f"\n================ {name.upper()} ================")

    prompt_map = {
        "translation": "Translate English to French",
        "synonym": "Find synonym",
        "antonym": "Find antonym"
    }

    results = {}

    baseline_correct = 0

    for ex in data:
        prompt = f"{prompt_map[name]}: {ex['input']}"
        pred = generate(model, tokenizer, prompt, device)
        if match(pred, ex["output"]):
            baseline_correct += 1

    baseline = baseline_correct / len(data)
    results["baseline"] = baseline

    print("baseline:", baseline)

    alphas = [0.05, 0.1, 0.2, 0.5]

    for lname, layer_idx in layers.items():

        results[lname] = {}

        print(f"\n--- {lname} ---")

        for a in alphas:

            acc = evaluate(
                model, tokenizer, data,
                device, layer_idx, vec, a,
                prompt_map[name]
            )

            results[lname][str(a)] = acc

            print(f"alpha={a}: {acc:.4f}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--subset", type=float, default=0.1)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.eval()

    # -----------------------------
    # DATASETS
    # -----------------------------
    translation = load_json("dataset_files/english-french.json", args.subset)
    synonym = load_json("dataset_files/synonym.json", args.subset)
    antonym = load_json("dataset_files/antonym.json", args.subset)

    L = len(model.model.layers)
    layers = get_layers(L)

    # -----------------------------
    # VECTOR (FROM TRANSLATION ONLY)
    # -----------------------------
    vec = build_vector(model, tokenizer, translation, device)

    all_results = {}

    all_results["translation"] = run_task(
        model, tokenizer, device,
        translation, "translation",
        vec, layers
    )

    all_results["synonym"] = run_task(
        model, tokenizer, device,
        synonym, "synonym",
        vec, layers
    )

    all_results["antonym"] = run_task(
        model, tokenizer, device,
        antonym, "antonym",
        vec, layers
    )

    # -----------------------------
    # SAVE
    # -----------------------------
    with open("results_fv.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nDONE → results_fv.json")


if __name__ == "__main__":
    main()