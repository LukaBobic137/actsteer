import os
import sys
import json
import pandas as pd
import tqdm
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir
os.chdir(project_dir)
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate
from compute_representations_fv import load_fv_dataset

config_path = os.path.join(project_dir, "config/format")


# -----------------------------
# RELAXED EVAL (FIX)
# -----------------------------
def normalize(s: str) -> str:
    return s.lower().strip().strip(".,;:!?\"'")


def relaxed_match(gen: str, expected: str) -> bool:
    gen = normalize(gen)
    expected = normalize(expected)
    return expected in gen or expected == gen


# -----------------------------
# PROMPT
# -----------------------------
def build_prompt(r, tokenizer):
    messages = [{"role": "user", "content": r["input"]}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )


# -----------------------------
# STEERING HOOK
# -----------------------------
def add_steering_hook(model, layer_idx, steering_vector, alpha=1.0):
    def hook_fn(module, input, output):
        h = output[0]

        # FIX dtype mismatch (VERY IMPORTANT)
        h = h + alpha * steering_vector.to(device=h.device, dtype=h.dtype)

        return (h,) + output[1:]

    return model.model.layers[layer_idx].register_forward_hook(hook_fn)


# -----------------------------
# MAIN
# -----------------------------
@hydra.main(config_path=config_path, config_name="compute_representations_fv")
def run(args: DictConfig):

    print(OmegaConf.to_yaml(args))

    # -------------------------
    # MODEL
    # -------------------------
    model, tokenizer = load_model_from_tl_name(
        args.model_name,
        device=args.device,
        cache_dir=args.transformers_cache_dir
    )
    model.eval()

    # -------------------------
    # DATA
    # -------------------------
    dfs = []
    for task in args.tasks:
        dfs.append(load_fv_dataset(task, args.fv_local_data_dir))

    data = pd.concat(dfs).reset_index(drop=True)

    if args.get("use_data_subset", False):
        ratio = float(args.get("data_subset_ratio", 0.1))
        data = data.head(max(1, int(len(data) * ratio)))

    # -------------------------
    # STEERING VECTOR
    # -------------------------
    hidden_size = model.config.hidden_size
    steering_vector = torch.zeros(hidden_size)
    steering_vector[:10] = 1.0
    steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)

    num_layers = model.config.num_hidden_layers

    steer_layers = {
        "baseline": None,
        "l25": int(num_layers * 0.25),
        "l50": int(num_layers * 0.50),
        "l75": int(num_layers * 0.75),
    }

    print("Steering layers:", steer_layers)

    # -------------------------
    # LOOP
    # -------------------------
    results = []
    pbar = tqdm.tqdm(total=len(data))

    for _, r in data.iterrows():

        prompt = build_prompt(r, tokenizer)

        row = {
            "input": r["input"],
            "target": r["output"]
        }

        # ---------------- BASELINE ----------------
        base = generate(
            model, tokenizer, prompt, args.device,
            max_new_tokens=args.max_generation_length
        )
        row["baseline"] = base

        # ---------------- STEERING ----------------
        for label, layer in steer_layers.items():

            if layer is None:
                continue

            handle = add_steering_hook(
                model,
                layer_idx=layer,
                steering_vector=steering_vector,
                alpha=1.0
            )

            out = generate(
                model, tokenizer, prompt, args.device,
                max_new_tokens=args.max_generation_length
            )

            handle.remove()

            row[f"{label}_output"] = out

        results.append(row)
        pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)

    # -------------------------
    # RELAXED EVAL (FIX)
    # -------------------------
    def eval_col(col):
        return df.apply(lambda r: relaxed_match(r[col], r["target"]), axis=1).mean()

    print("\n=== Accuracy (relaxed) ===")
    print("baseline:", eval_col("baseline"))

    for c in ["l25_output", "l50_output", "l75_output"]:
        print(f"{c}:", eval_col(c))

    # -------------------------
    # SAVE (NO HDF5 BUG)
    # -------------------------
    if not args.dry_run:
        out_dir = os.path.join(
            script_dir,
            "steering_results",
            args.model_name.replace("/", "_")
        )
        os.makedirs(out_dir, exist_ok=True)

        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

        print("Saved →", out_dir)


if __name__ == "__main__":
    run()