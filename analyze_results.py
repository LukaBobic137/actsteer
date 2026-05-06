import json
import pandas as pd

PATH = "steering_results/meta-llama_Meta-Llama-3-8B-Instruct/results_fv.json"

# -----------------------------
# load JSON
# -----------------------------
with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())


# -----------------------------
# helper (robust normalization)
# -----------------------------
def normalize(x):
    if x is None:
        return ""
    return str(x).strip().lower().split("\n")[0]


def match(pred, target):
    pred = normalize(pred)
    target = normalize(target)

    if pred == "" or target == "":
        return False

    # take first token (FV tasks are word-level)
    return pred.split()[0] == target.split()[0]


# -----------------------------
# accuracy per setting
# -----------------------------
cols = ["baseline", "l25_output", "l50_output", "l75_output"]

print("\n=== Accuracy ===")
for c in cols:
    acc = df.apply(lambda r: match(r[c], r["target"]), axis=1).mean()
    print(f"{c}: {acc:.3f}")


# -----------------------------
# steering effect (change rate)
# -----------------------------
print("\n=== Change rate vs baseline ===")
for c in cols[1:]:
    change = (df["baseline"] != df[c]).mean()
    print(f"{c}: {change:.3f}")


# -----------------------------
# sanity check
# -----------------------------
print("\n=== Sample ===")
print(df[["input", "target", "baseline", "l25_output"]].head(5))