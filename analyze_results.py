import json

PATH = "results_fv.json"

with open(PATH, "r") as f:
    data = json.load(f)


def analyze_task(task_name, res):
    print(f"\n================ {task_name.upper()} ================")

    baseline = res["baseline"]
    print(f"baseline: {baseline:.4f}")

    best_layer = None
    best_value = baseline

    print("\n--- layers ---")
    for k, v in res.items():
        if k == "baseline":
            continue

        delta = v - baseline
        print(f"{k}: {v:.4f} (Δ {delta:+.4f})")

        if v > best_value:
            best_value = v
            best_layer = k

    # -----------------------------
    # SUMMARY
    # -----------------------------
    print("\n--- summary ---")

    if best_layer is None:
        print("❌ No improvement over baseline")
    else:
        print(f"✅ Best layer: {best_layer} (+{best_value - baseline:.4f})")

    return {
        "baseline": baseline,
        "best_layer": best_layer,
        "best_gain": best_value - baseline
    }


# -----------------------------
# RUN ANALYSIS
# -----------------------------
summary = {}

for task, res in data.items():
    summary[task] = analyze_task(task, res)


# -----------------------------
# CROSS-TASK INTERPRETATION
# -----------------------------
print("\n================ GLOBAL INTERPRETATION ================")

t = summary.get("translation", {})
s = summary.get("synonym", {})
a = summary.get("antonym", {})

if t.get("best_gain", 0) > 0:
    print("✔ Steering improves TRANSLATION → signal exists")
else:
    print("✖ No improvement on TRANSLATION → weak vector")

if s.get("best_gain", 0) > 0:
    print("✔ Generalizes to SYNONYMS → semantic direction")
else:
    print("• No synonym gain → task-specific vector")

if a.get("best_gain", 0) < 0:
    print("✔ ANTONYM drop → expected (semantic bias)")
elif a.get("best_gain", 0) > 0:
    print("⚠ Unexpected ANTONYM improvement → check behavior")
else:
    print("• Neutral on ANTONYM")


# -----------------------------
# FINAL SCORE
# -----------------------------
print("\n================ QUICK SCORE ================")

score = 0

if t.get("best_gain", 0) > 0:
    score += 1
if s.get("best_gain", 0) > 0:
    score += 1
if a.get("best_gain", 0) >= -0.01:  # toleriramo mali pad
    score += 1

print(f"Score: {score}/3")

if score == 3:
    print("🔥 Excellent steering vector")
elif score == 2:
    print("👍 Good, but can be improved")
elif score == 1:
    print("⚠ Weak signal")
else:
    print("❌ Steering not working")