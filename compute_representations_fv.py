# %%
import os
import sys
import pandas as pd
import tqdm
import json
import requests
import numpy as np
from typing import Optional
from omegaconf import DictConfig, OmegaConf
import hydra

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir   # script is in the project root
os.chdir(project_dir)
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate, extract_representation

config_path = os.path.join(project_dir, 'config/format')

# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

# Raw GitHub base URL for the Function Vectors dataset files
FV_DATASET_BASE = "https://raw.githubusercontent.com/ericwtodd/function_vectors/master/dataset_files"

# Datasets from Appendix E that we care about.
# Each entry maps a short task name → the JSON filename in the FV repo.
FV_DATASETS = {
    "english_french":    "abstractive/english_french.json",
    "english_german":    "abstractive/english_german.json",
    "synonyms":          "abstractive/synonym.json",
    "antonyms":          "abstractive/antonym.json",
    "country_capital":   "extractive/country_capital.json",
    "person_sport":      "extractive/person_sport.json",
}


def load_fv_dataset(task_name: str, local_data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load a Function Vectors dataset.

    Tries a local directory first (useful if you've already cloned the repo),
    then falls back to downloading directly from GitHub.

    Each JSON file in the FV repo has the structure:
        {"input": [...], "output": [...]}
    where input[i] / output[i] are the i-th example pair.

    Returns a DataFrame with columns: task, input, output.
    """
    filename = FV_DATASETS[task_name]

    data = None

    # 1. Try local file
    if local_data_dir:
        local_path = os.path.join(local_data_dir, filename)
        if os.path.exists(local_path):
            with open(local_path) as f:
                data = json.load(f)

    # 2. Fall back to GitHub
    if data is None:
        url = f"{FV_DATASET_BASE}/{filename}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

    inputs  = data["input"]
    outputs = data["output"]
    assert len(inputs) == len(outputs), "Mismatch between inputs and outputs"

    df = pd.DataFrame({"input": inputs, "output": outputs})
    df.insert(0, "task", task_name)
    return df


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

# Zero-shot prompt templates for each task
PROMPT_TEMPLATES = {
    "english_french":  "Translate the following English word to French.\nEnglish: {input}\nFrench:",
    "english_german":  "Translate the following English word to German.\nEnglish: {input}\nGerman:",
    "synonyms":        "Give a synonym for the following word.\nWord: {input}\nSynonym:",
    "antonyms":        "Give an antonym for the following word.\nWord: {input}\nAntonym:",
    "country_capital": "What is the capital city of the following country?\nCountry: {input}\nCapital:",
    "person_sport":    "What sport is the following person known for?\nPerson: {input}\nSport:",
}

# Few-shot (in-context) variant — same task but with k examples prepended.
# We build these dynamically in build_few_shot_prompt().


def build_zero_shot_prompt(task_name: str, input_word: str) -> str:
    return PROMPT_TEMPLATES[task_name].format(input=input_word)


def build_few_shot_prompt(
    task_name: str,
    input_word: str,
    few_shot_examples: list,   # list of {"input": ..., "output": ...}
) -> str:
    """Build a k-shot prompt by prepending examples to the zero-shot template."""
    template = PROMPT_TEMPLATES[task_name]
    # Split template into the "context" part and the "query" part
    # Everything before the last newline is the instruction header.
    lines = template.split("\n")
    header   = "\n".join(lines[:-1])   # e.g. "Translate English to French.\nEnglish: {input}\nFrench:"
    response_prefix = lines[-1]        # e.g. "French:"

    # Build few-shot block
    shot_lines = []
    for ex in few_shot_examples:
        shot_lines.append(header.format(input=ex["input"]) + " " + ex["output"])

    # Final query
    shot_lines.append(header.format(input=input_word))
    shot_lines.append(response_prefix)    # empty response prefix for model to complete

    return "\n".join(shot_lines)


# ---------------------------------------------------------------------------
# Layer-index helpers
# ---------------------------------------------------------------------------

def get_depth_layer_indices(num_layers: int) -> dict:
    """Return layer indices at 25 %, 50 %, and 75 % of total model depth."""
    return {
        "layer_25pct": max(0, round(num_layers * 0.25) - 1),
        "layer_50pct": max(0, round(num_layers * 0.50) - 1),
        "layer_75pct": max(0, round(num_layers * 0.75) - 1),
    }


# ---------------------------------------------------------------------------
# Main hydra entrypoint
# ---------------------------------------------------------------------------

@hydra.main(config_path=config_path, config_name='compute_representations_fv')
def compute_representations_fv(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    # ---- Load model --------------------------------------------------------
    model_name = args.model_name
    model, tokenizer = load_model_from_tl_name(
        model_name, device=args.device,
        cache_dir=args.transformers_cache_dir,
    )
    # NOTE: do NOT call model.to(device) here — load_model_from_tl_name uses
    # device_map="auto" which already places layers on the correct devices.
    model.eval()

    # Llama 3 stores depth at model.config.num_hidden_layers
    num_layers = getattr(model.config, "num_hidden_layers", None) \
              or getattr(model.config, "n_layer", None) \
              or getattr(model.config, "num_layers", None)
    if num_layers is None:
        raise RuntimeError(
            "Cannot determine model depth from model.config. "
            "Set num_layers manually in the script."
        )
    num_layers = int(num_layers)

    depth_layers = get_depth_layer_indices(num_layers)
    print(f"Model depth: {num_layers} layers")
    print(f"Probing layers: {depth_layers}")

    # ---- Determine which tasks to run -------------------------------------
    tasks_to_run = list(args.get("tasks", FV_DATASETS.keys()))

    # ---- Process each task -------------------------------------------------
    for task_name in tasks_to_run:
        print(f"\n{'='*60}\nProcessing task: {task_name}\n{'='*60}")

        # Load dataset
        local_dir = args.get("fv_local_data_dir", None)
        task_df = load_fv_dataset(task_name, local_dir)

        if args.get("use_data_subset", False):
            ratio = float(args.get("data_subset_ratio", 0.1))
            task_df = task_df.iloc[:max(1, int(len(task_df) * ratio))]

        if args.get("dry_run", False):
            task_df = task_df.head(5)

        task_df.reset_index(drop=True, inplace=True)

        # Optional few-shot examples (taken from the task itself, held out
        # from the main loop so they're never evaluated)
        num_shots = int(args.get("num_shots", 0))
        few_shot_pool = task_df.head(num_shots).to_dict("records") if num_shots > 0 else []
        eval_df = task_df.iloc[num_shots:].reset_index(drop=True)

        rows = []
        p_bar = tqdm.tqdm(total=len(eval_df), desc=task_name)

        for _, r in eval_df.iterrows():
            row = dict(r)
            input_word = row["input"]
            expected    = row["output"]

            # ---- Build prompts -------------------------------------------
            if num_shots > 0:
                prompt = build_few_shot_prompt(task_name, input_word, few_shot_pool)
            else:
                prompt = build_zero_shot_prompt(task_name, input_word)

            # Use chat template for instruct models, raw prompt for base models
            is_instruct = "instruct" in model_name.lower() or "-it" in model_name.lower()
            if is_instruct:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )
            else:
                # Base model — plain text prompt
                formatted_prompt = prompt

            # ---- Generate output ------------------------------------------
            generated = generate(
                model, tokenizer, formatted_prompt, args.device,
                max_new_tokens=args.get("max_generation_length", 20),
            )

            # ---- Extract representations at 25 / 50 / 75 % depth ----------
            layer_reps = {}
            for depth_label, layer_idx in depth_layers.items():
                rep = extract_representation(
                    model, tokenizer, formatted_prompt, args.device,
                    num_final_tokens=int(args.get("num_final_tokens", 1)),
                    layer_idx=layer_idx,
                )
                layer_reps[depth_label] = rep

            # ---- Store results --------------------------------------------
            row["formatted_prompt"] = formatted_prompt
            row["generated_output"] = generated
            row["expected_output"]  = expected

            # Store as numpy arrays keyed by depth label
            for depth_label, rep in layer_reps.items():
                row[f"repr_{depth_label}"] = rep

            rows.append(row)
            p_bar.update(1)

        p_bar.close()
        result_df = pd.DataFrame(rows)

        # ---- Quick accuracy estimate (exact-match on first token) ----------
        def first_token_match(gen: str, expected: str) -> bool:
            gen_tok = gen.strip().split()[0].lower().strip(".,;:!?") if gen.strip() else ""
            exp_tok = expected.strip().split()[0].lower().strip(".,;:!?")
            return gen_tok == exp_tok

        result_df["correct"] = result_df.apply(
            lambda r: first_token_match(r["generated_output"], r["expected_output"]), axis=1
        )
        acc = result_df["correct"].mean()
        print(f"\nTask {task_name} — first-token accuracy: {acc:.3f}")

        # ---- Save ----------------------------------------------------------
        if not args.get("dry_run", False):
            folder = os.path.join(
                script_dir, "representations_fv", model_name,
                f"shots_{num_shots}",
            )
            os.makedirs(folder, exist_ok=True)

            safe_task = task_name.replace(":", "_")
            out_path = os.path.join(folder, f"{safe_task}.h5")
            result_df.to_hdf(out_path, key="df", mode="w")
            print(f"Saved → {out_path}")

            # Also save a lightweight CSV (without the large numpy arrays)
            csv_df = result_df.drop(
                columns=[c for c in result_df.columns if c.startswith("repr_")]
            )
            csv_df.to_csv(out_path.replace(".h5", "_text.csv"), index=False)


# %%
if __name__ == '__main__':
    compute_representations_fv()
# %%