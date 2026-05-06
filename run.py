import os
import sys
import json
import pandas as pd
import tqdm
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate

# ------------------------------------------------------------
# setup
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir
os.chdir(project_dir)
sys.path.append(project_dir)

config_path = os.path.join(project_dir, "config/format")


# ------------------------------------------------------------
# prompt templates (FIXED: constrained outputs)
# ------------------------------------------------------------
PROMPT_TEMPLATES = {
    "english_french":
        "Translate to French.\nReturn ONLY ONE WORD.\nEnglish: {input}\nFrench:",

    "english_german":
        "Translate to German.\nReturn ONLY ONE WORD.\nEnglish: {input}\nGerman:",

    "synonyms":
        "Give ONE synonym.\nReturn ONLY ONE WORD.\nWord: {input}\nSynonym:",

    "antonyms":
        "Give ONE antonym.\nReturn ONLY ONE WORD.\nWord: {input}\nAntonym:",
}


# ------------------------------------------------------------
# utils
# ------------------------------------------------------------
def normalize(x):
    return str(x).strip().lower()


def exact_match(pred, target):
    if not isinstance(pred, str):
        return False
    return normalize(pred) == normalize(target)


def build_prompt(r, tokenizer, task):
    messages = [{"role": "user", "content": PROMPT_TEMPLATES[task].format(input=r["input"])}]
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
        h = h + alpha * steering_vector.to(h.device)
        return (h,) + output[1:]

    return model.model.layers[layer_idx].register_forward_hook(hook_fn)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
@hydra.main(config_path=config_path, config_name="compute_representations_fv")
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
    # load data
    # -----------------------
    local_dir = args.get("fv_local_data_dir", None)

    dfs = []
    for task in args.tasks:
        df = load_fv_dataset(task, local_dir)
        df["task"] = task
        dfs.append(df)

    data = pd.concat(dfs)

    # shuffle (IMPORTANT FIX)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # subset / dry run
    if args.get("use_data_subset", False):
        data = data.head(int(len(data) * args.data_subset_ratio))

    if args.get("dry_run", False):
        data = data.head(5)

    # -----------------------
    # steering vector (placeholder)
    # -----------------------
    hidden_size = model.config.hidden_size
    steering_vector = torch.zeros(hidden_size)
    steering_vector[:10] = 1.0
    steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)

    layers = model.config.num_hidden_layers
    steer_layers = {
        "baseline": None,
        "l25": int(layers * 0.25),
        "l50": int(layers * 0.50),
        "l75": int(layers * 0.75),
    }

    print("Steering layers:", steer_layers)

    # -----------------------
    # run
    # -----------------------
    results = []
    pbar = tqdm.tqdm(total=len(data))

    for _, r in data.iterrows():

        task = r["task"]
        prompt = build_prompt(r, tokenizer, task)

        row = {
            "input": r["input"],
            "target": r["output"],
            "task": task
        }

        # -----------------------
        # baseline
        # -----------------------
        baseline = generate(
            model, tokenizer, prompt, args.device,
            max_new_tokens=args.max_generation_length
        )
        row["baseline"] = baseline

        # -----------------------
        # steering runs
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

            row[label] = out

        results.append(row)
        pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)

    # -----------------------
    # evaluation (FIXED)
    # -----------------------
    print("\n=== Accuracy ===")
    for col in ["baseline", "l25", "l50", "l75"]:
        acc = df.apply(lambda r: exact_match(r[col], r["target"]), axis=1).mean()
        print(f"{col}: {acc:.3f}")

    print("\n=== Change rate vs baseline ===")
    for col in ["l25", "l50", "l75"]:
        change = (df["baseline"] != df[col]).mean()
        print(f"{col}: {change:.3f}")

    # -----------------------
    # save
    # -----------------------
    if not args.get("dry_run", False):

        out_dir = os.path.join(
            script_dir,
            "steering_results",
            args.model_name
        )
        os.makedirs(out_dir, exist_ok=True)

        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

        print("\nSaved →", out_dir)


# ------------------------------------------------------------
if __name__ == "__main__":
    run()