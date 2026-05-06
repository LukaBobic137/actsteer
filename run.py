import os
import sys
import pandas as pd
import tqdm
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from compute_representations_fv import load_fv_dataset

# ------------------------------------------------------------
# setup
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir
os.chdir(project_dir)
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate

# ------------------------------------------------------------
# FV DATASET
# ------------------------------------------------------------
FV_DATASET_BASE = "https://raw.githubusercontent.com/ericwtodd/function_vectors/master/dataset_files"

FV_DATASETS = {
    "english_french": "abstractive/english-french.json",
    "english_german": "abstractive/english-german.json",
}

def load_fv_dataset(task_name, local_dir=None):
    import json, requests

    filename = FV_DATASETS[task_name]

    data = None

    if local_dir:
        path = os.path.join(local_dir, filename)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

    if data is None:
        url = f"{FV_DATASET_BASE}/{filename}"
        data = requests.get(url, timeout=30).json()

    if isinstance(data, list):
        inputs = [x["input"] for x in data]
        outputs = [x["output"] for x in data]
    else:
        inputs = data["input"]
        outputs = data["output"]

    return pd.DataFrame({"input": inputs, "output": outputs})


# ------------------------------------------------------------
# prompt
# ------------------------------------------------------------
def build_prompt(r, tokenizer):
    messages = [{"role": "user", "content": r["input"]}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )


# ------------------------------------------------------------
# steering hook
# ------------------------------------------------------------
def add_steering_hook(model, layer_idx, steering_vector, alpha=1.0):

    def hook_fn(module, input, output):
        h = output[0]
        h = h + alpha * steering_vector.to(h.device).to(h.dtype)
        return (h,) + output[1:]

    return model.model.layers[layer_idx].register_forward_hook(hook_fn)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
@hydra.main(config_path="config/format", config_name="compute_representations_fv")
def run(args: DictConfig):

    print(OmegaConf.to_yaml(args))

    # ----------------------------
    # model
    # ----------------------------
    model, tokenizer = load_model_from_tl_name(
        args.model_name,
        device=args.device,
        cache_dir=args.transformers_cache_dir
    )
    model.eval()

    model_dtype = next(model.parameters()).dtype

    # ----------------------------
    # data
    # ----------------------------
    tasks = args.tasks if isinstance(args.tasks, list) else [args.tasks]

    dfs = []
    for t in tasks:
        dfs.append(load_fv_dataset(t, args.fv_local_data_dir))

    data = pd.concat(dfs, ignore_index=True)

    # subset
    if args.use_data_subset:
        data = data.sample(frac=float(args.data_subset_ratio), random_state=42)

    if args.dry_run:
        data = data.head(5)

    # ----------------------------
    # steering vector (PLACEHOLDER)
    # ----------------------------
    hidden = model.config.hidden_size

    steering_vector = torch.zeros(hidden, dtype=model_dtype)
    steering_vector[:10] = 1.0
    steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)

    # layer selection
    n_layers = model.config.num_hidden_layers

    steer_layers = {
        "baseline": None,
        "l25": int(n_layers * 0.25),
        "l50": int(n_layers * 0.50),
        "l75": int(n_layers * 0.75),
    }

    print("Steering layers:", steer_layers)

    # ----------------------------
    # eval loop
    # ----------------------------
    results = []

    for _, r in tqdm.tqdm(data.iterrows(), total=len(data)):

        prompt = build_prompt(r, tokenizer)

        row = {
            "input": r["input"],
            "target": r["output"]
        }

        # baseline
        row["baseline"] = generate(model, tokenizer, prompt, args.device)

        # steering runs
        for label, layer in steer_layers.items():

            if layer is None:
                continue

            handle = add_steering_hook(
                model,
                layer,
                steering_vector,
                alpha=1.0
            )

            out = generate(model, tokenizer, prompt, args.device)

            handle.remove()

            row[label] = out

        results.append(row)

    df = pd.DataFrame(results)

    # ----------------------------
    # save
    # ----------------------------
    if not args.dry_run:
        out_dir = os.path.join(
            "steering_results",
            args.model_name.replace("/", "_")
        )
        os.makedirs(out_dir, exist_ok=True)

        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

        print("Saved →", out_dir)


if __name__ == "__main__":
    run()