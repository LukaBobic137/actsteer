import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
N_SAMPLES = 100
ALPHA = 1.0

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

# fix warning
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD DATASET
# =========================
print("Loading dataset...")

# ⚠️ OVDJE PROMIJENI PUT DO JSON-a
dataset = load_dataset(
    "json",
    data_files="function_vectors/dataset_files/translation/en_de.json",
    split="train"
)

dataset = dataset.select(range(N_SAMPLES))

print("Columns:", dataset.column_names)
print("Example:", dataset[0])

# =========================
# HELPERS
# =========================
def build_prompt(word):
    return f"Translate the following English word to German:\n{word}\nAnswer:"

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def is_correct(pred, true):
    return true.lower() in pred.lower()

# =========================
# LAYER SETUP (25/50/75)
# =========================
n_layers = len(model.model.layers)

layer_ids = [
    int(0.25 * n_layers),
    int(0.50 * n_layers),
    int(0.75 * n_layers),
]

print("Using layers:", layer_ids)

# =========================
# MAIN LOOP
# =========================
results = {}

for LAYER_ID in layer_ids:
    print(f"\n=== LAYER {LAYER_ID} ===")

    correct_acts = []
    wrong_acts = []

    def hook_fn(module, input, output):
        acts.append(output[:, -1, :].detach().cpu())

    handle = model.model.layers[LAYER_ID].mlp.register_forward_hook(hook_fn)

    print("Collecting activations...")

    for example in tqdm(dataset):
        word = example["input"]
        true_answer = example["output"]

        prompt = build_prompt(word)

        acts = []

        _ = generate(prompt)

        if len(acts) == 0:
            continue

        act = acts[0]

        pred = generate(prompt)

        if is_correct(pred, true_answer):
            correct_acts.append(act)
        else:
            wrong_acts.append(act)

    handle.remove()

    print(f"Correct: {len(correct_acts)}, Wrong: {len(wrong_acts)}")

    if len(correct_acts) == 0 or len(wrong_acts) == 0:
        print("Skipping layer (no balance)")
        continue

    # =========================
    # BUILD STEERING VECTOR
    # =========================
    print("Building steering vector...")

    correct_mean = torch.stack(correct_acts).mean(dim=0)
    wrong_mean = torch.stack(wrong_acts).mean(dim=0)

    steering_vector = correct_mean - wrong_mean

    # =========================
    # EVALUATION
    # =========================
    def evaluate(use_steering=False):
        if use_steering:
            def steering_hook(module, input, output):
                return output + ALPHA * steering_vector.to(output.device)

            h = model.model.layers[LAYER_ID].mlp.register_forward_hook(steering_hook)

        correct = 0
        total = 0

        for example in dataset:
            word = example["input"]
            true_answer = example["output"]

            prompt = build_prompt(word)

            pred = generate(prompt)

            if is_correct(pred, true_answer):
                correct += 1

            total += 1

        if use_steering:
            h.remove()

        return correct / total

    print("Evaluating baseline...")
    acc_base = evaluate(False)

    print("Evaluating with steering...")
    acc_steer = evaluate(True)

    results[LAYER_ID] = (acc_base, acc_steer)

# =========================
# FINAL RESULTS
# =========================
print("\nFINAL RESULTS:")

for layer, (base, steer) in results.items():
    print(f"Layer {layer}: base={base:.3f}, steer={steer:.3f}")