import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
# =========================
# CONFIG
# =========================
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
LAYER_ID = 10
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

# =========================
# LOAD DATASET
# =========================
print("Loading dataset...")

ds = load_dataset(
    "fblgit/simple-math",
    split="train",
    verification_mode="no_checks"
)

dataset = ds.select(range(N_SAMPLES))
# =========================
# HELPERS
# =========================
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_number(text):
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None

def is_correct(pred, true):
    return extract_number(pred) == str(true)

# =========================
# COLLECT ACTIVATIONS
# =========================
print("Collecting activations...")

correct_acts = []
wrong_acts = []

def hook_fn(module, input, output):
    acts.append(output[:, -1, :].detach().cpu())

handle = model.model.layers[LAYER_ID].mlp.register_forward_hook(hook_fn)

for example in tqdm(dataset):
    prompt = example["input"]
    true_answer = example["output"]

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

# =========================
# BUILD STEERING VECTOR
# =========================
print("Building steering vector...")

correct_mean = torch.stack(correct_acts).mean(dim=0)
wrong_mean = torch.stack(wrong_acts).mean(dim=0)

steering_vector = correct_mean - wrong_mean

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(use_steering=False):
    if use_steering:
        def steering_hook(module, input, output):
            return output + ALPHA * steering_vector.to(output.device)

        handle = model.model.layers[LAYER_ID].mlp.register_forward_hook(steering_hook)

    correct = 0
    total = 0

    for example in dataset:
        prompt = example["input"]
        true_answer = example["output"]

        pred = generate(prompt)

        if is_correct(pred, true_answer):
            correct += 1
        total += 1

    if use_steering:
        handle.remove()

    return correct / total

# =========================
# RUN EVAL
# =========================
print("\nEvaluating baseline...")
acc_base = evaluate(use_steering=False)

print("Evaluating with steering...")
acc_steer = evaluate(use_steering=True)

print("\nRESULTS:")
print(f"Baseline accuracy: {acc_base:.3f}")
print(f"Steered accuracy:  {acc_steer:.3f}")