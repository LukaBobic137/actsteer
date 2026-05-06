import os
import sys
import pandas as pd
import tqdm
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

# IMPORTANT: import iz compute_representations_fv
from compute_representations_fv import load_fv_dataset

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir
os.chdir(project_dir)
sys.path.append(project_dir)

config_path = os.path.join(project_dir, "config/format")


# -------------------------
# prompt
# -------------------------
def build_prompt(r, tokenizer):
    messages = [{"role": "user", "content": r["input"]}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )


# -------------------------
# steering hook
# -------------------------
def add_steering_hook(model, layer_idx, steering_vector, alpha=1.0):

    def hook_fn(module, input, output):
        h = output[0]
        h = h + alpha * steering_vector.to(h.device)
        return (h,) + output[1:]

    return model.model.layers[layer_idx].register_forward_hook(hook_fn)


# -------------------------
# MAIN
# -------------------------
@hydra.main(config_path=config_path, config_name="compute_representations_fv")
def run(args: DictConfig):

    print(OmegaConf.to_yaml(args))

    # ---------------- model ----------------
    model, tokenizer = load_model_from_tl_name(
        args.model_name,
        device=args.device,
        cache_dir=args.transformers_cache_dir
    )

    model.eval()

    # force dtype fix (IMPORTANT for tvoj error)
    model = model.to(torch.float16)

    # ---------------- dataset ----------------
    tasks = args.tasks
    if isinstance(tasks, str):
        tasks = [tasks]

    dfs = []
    for t in tasks:
        dfs.append(load_fv_dataset(t, args.fv_local_data_dir))

    data = pd.concat(dfs).reset_index(drop=True)

    if args.use_data_subset:
        data = data.sample(frac=float(args.data_subset_ratio), random_state=42)

    if args.dry_run:
        data = data.head(5)

    # ---------------- steering setup ----------------
    hidden_size = model.config.hidden_size

    steering_vector = torch.zeros(hidden_size, dtype=torch.float16)
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

    # ---------------- loop ----------------
    results = []
    pbar = tqdm.tqdm(total=len(data))

    for _, r in data.iterrows():

        prompt = build_prompt(r, tokenizer)

        row = {
            "input": r["input"],
            "target": r["output"]
        }

        # baseline
        row["baseline"] = generate(
            model, tokenizer, prompt, args.device,
            max_new_tokens=args.max_generation_length
        )

        # steering runs
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

    # ---------------- save ----------------
    if not args.dry_run:
        out_dir = os.path.join(script_dir, "steering_results")
        os.makedirs(out_dir, exist_ok=True)

        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

        print("Saved →", out_dir)


if __name__ == "__main__":
    run()