import pandas as pd

PATH = "steering_results/meta-llama_Meta-Llama-3-8B-Instruct/results.csv"

df = pd.read_csv(PATH)

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# -----------------------------
# helper
# -----------------------------
def match(pred, target):
    if not isinstance(pred, str):
        return False
    pred_tok = pred.strip().split()[0].lower()
    target_tok = str(target).strip().split()[0].lower()
    return pred_tok == target_tok


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
# quick inspection
# -----------------------------
print("\n=== Sample ===")
print(df[["input", "target", "baseline", "l25_output"]].head(3))