import os
import sys
import json
import pandas as pd
import tqdm
import torch

from omegaconf import DictConfig, OmegaConf
import hydra

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir
os.chdir(project_dir)
sys.path.append(project_dir)

config_path = os.path.join(project_dir, "config/format")


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
def load_data(path, limit=None):
    with open(path) as f:
        data = [json.loads(l) for l in f.readlines()]
    df = pd.DataFrame(data)
    if limit:
        df = df.head(limit)
    return df


# ------------------------------------------------------------
# PROMPT
# ------------------------------------------------------------
def build_prompt(r, tokenizer):
    messages = [{"role": "user", "content": r["prompt"]}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )


# ------------------------------------------------------------
# STEERING HOOK
# ------------------------------------------------------------
def add_steering_hook(model, layer_idx, steering_vector, alpha=1.0):

    def hook_fn(module, input, output):
        h = output[0]
        h = h + alpha * steering_vector.to(h.device)
        return (h,) + output[1:]

    return model.model.layers[layer_idx].register_forward_hook(hook_fn)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
@hydra.main(config_path=config_path, config_name="compute_representations")
def run(args: DictConfig):

    print(OmegaConf.to_yaml(args))

    # -----------------------
    # model
    # -----------------------
    model, tokenizer = load_model_from_tl_name(
        args.model_name,
        device=args.device,
        cache_dir=args.transformers_cache_dir
    )
    model.eval()

    # -----------------------
    # data
    # -----------------------
    data = load_data(args.data_path, limit=5 if args.dry_run else None)

    # -----------------------
    # STEERING SETUP
    # -----------------------
    hidden_size = model.config.hidden_size

    # ⚠️ PLACEHOLDER VECTOR (replace with real one later)
    steering_vector = torch.zeros(hidden_size)
    steering_vector[:10] = 1.0
    steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)

    steer_layers = {
        "baseline": None,
        "l25": int(model.config.num_hidden_layers * 0.25),
        "l50": int(model.config.num_hidden_layers * 0.50),
        "l75": int(model.config.num_hidden_layers * 0.75),
    }

    print("Steering layers:", steer_layers)

    # -----------------------
    # LOOP
    # -----------------------
    results = []
    pbar = tqdm.tqdm(total=len(data))

    for _, r in data.iterrows():

        prompt = build_prompt(r, tokenizer)

        row = {
            "input": r.get("prompt", ""),
            "target": r.get("output", None),
        }

        # -----------------------
        # BASELINE
        # -----------------------
        base_out = generate(
            model, tokenizer, prompt, args.device,
            max_new_tokens=args.max_generation_length
        )
        row["baseline"] = base_out

        # -----------------------
        # STEERING RUNS
        # -----------------------
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

    # -----------------------
    # SAVE
    # -----------------------
    if not args.dry_run:
        out_dir = os.path.join(
            script_dir,
            "steering_results",
            args.model_name
        )
        os.makedirs(out_dir, exist_ok=True)

        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
        df.to_hdf(os.path.join(out_dir, "results.h5"), key="df", mode="w")

        print("Saved results →", out_dir)


# ------------------------------------------------------------
if __name__ == "__main__":
    run()