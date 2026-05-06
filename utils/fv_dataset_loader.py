import os
import json
from typing import List, Dict


SUPPORTED_TASKS = {
    "english-french": "english-french.json",
    "english-german": "english-german.json",
    "english-spanish": "english-spanish.json",
}


def load_fv_dataset(data_dir: str, tasks: List[str]):
    """
    Loads Function Vector abstractive datasets.

    Expected format:
    [
        {"input": "...", "output": "..."},
        ...
    ]
    """

    all_data = []

    for task in tasks:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unknown task: {task}")

        path = os.path.join(data_dir, SUPPORTED_TASKS[task])

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data:
            if "input" not in ex or "output" not in ex:
                continue

            all_data.append({
                "task": task,
                "input": ex["input"],
                "target": ex["output"]
            })

    return all_data