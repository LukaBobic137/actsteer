import os
import json
import pandas as pd


def load_fv_dataset(data_dir):

    files = [
        "english-french.json",
        "english-german.json",
        "english-spanish.json"
    ]

    all_data = []

    for f in files:
        path = os.path.join(data_dir, f)

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for item in data:
            all_data.append({
                "input": item["input"],
                "target": item["output"]
            })

    return pd.DataFrame(all_data)