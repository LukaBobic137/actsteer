import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DATA_PATH = "function_vectors/dataset_files/extractive/verb_v_adjective_3.json"

N_SAMPLES = 100
ALPHA = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

# =========================
# LOAD DATASET
# =========================
print("Loading dataset...")

with open(DATA_PATH, "r") as f:
    raw = json.load(f)

dataset = raw[:N_SAMPLES]

# =========================
# HELPERS
# =========================
def detect_keys(example):
    """Find input/output keys dynamically"""
    input_keys = ["input", "prompt", "instruction", "text"]
    output_keys = ["output", "answer", "target", "label"]

    inp = next((k for k in input_keys if k in example), None)
    out = next((k for k in output_keys if k in example), None)

    if inp is None or out is None:
        raise ValueError(f"Unknown format: {example.keys()}")

    return inp, out


def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def is_correct(pred, true):
    return str(true).strip().lower() in pred.strip().lower()


# =========================
# LAYER SELECTION (25%, 50%, 75%)
# =========================
n_layers = len(model.model.layers)

layer_ids = [
    n_layers // 4,
    n_layers // 2,
    (3 * n_layers) // 4
]

print(f"Using layers: {layer_ids}")

# =========================
# HOOK STORAGE
# =========================
activations = {lid: [] for lid in layer_ids}


def make_hook(layer_id):
    def hook(module, input, output):
        activations[layer_id].append(output[:, -1, :].detach().cpu())
    return hook


handles = []
for lid in layer_ids:
    h = model.model.layers[lid].mlp.register_forward_hook(make_hook(lid))
    handles.append(h)


# =========================
# COLLECT ACTIVATIONS
# =========================
print("Collecting activations...")

correct_acts = {lid: [] for lid in layer_ids}
wrong_acts = {lid: [] for lid in layer_ids}

for example in tqdm(dataset):
    inp_key, out_key = detect_keys(example)

    prompt = example[inp_key]
    true = example[out_key]

    for lid in layer_ids:
        activations[lid] = []

    pred = generate(prompt)

    for lid in layer_ids:
        if len(activations[lid]) == 0:
            continue

        act = activations[lid][0]

        if is_correct(pred, true):
            correct_acts[lid].append(act)
        else:
            wrong_acts[lid].append(act)


for h in handles:
    h.remove()


# =========================
# BUILD STEERING VECTORS
# =========================
print("Building steering vectors...")

steering_vectors = {}

for lid in layer_ids:
    if len(correct_acts[lid]) == 0 or len(wrong_acts[lid]) == 0:
        print(f"Skipping layer {lid} (not enough data)")
        continue

    c = torch.stack(correct_acts[lid]).mean(dim=0)
    w = torch.stack(wrong_acts[lid]).mean(dim=0)

    steering_vectors[lid] = c - w


# =========================
# EVALUATION
# =========================
def evaluate(use_steering=False):

    handles = []

    if use_steering:

        def make_steer_hook(lid):
            vec = steering_vectors[lid]

            def hook(module, input, output):
                return output + ALPHA * vec.to(output.device)

            return hook

        for lid in steering_vectors:
            h = model.model.layers[lid].mlp.register_forward_hook(
                make_steer_hook(lid)
            )
            handles.append(h)

    correct = 0
    total = 0

    for example in dataset:
        inp_key, out_key = detect_keys(example)

        prompt = example[inp_key]
        true = example[out_key]

        pred = generate(prompt)

        if is_correct(pred, true):
            correct += 1
        total += 1

    for h in handles:
        h.remove()

    return correct / total


# =========================
# RUN
# =========================
print("\nEvaluating baseline...")
acc_base = evaluate(use_steering=False)

print("Evaluating with steering...")
acc_steer = evaluate(use_steering=True)

print("\nRESULTS:")
print(f"Baseline accuracy: {acc_base:.3f}")
print(f"Steered accuracy:  {acc_steer:.3f}")