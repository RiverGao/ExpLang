from datasets import load_dataset
import pandas as pd
import os
import argparse
import numpy as np


np.random.seed(42)

parser = argparse.ArgumentParser(description="convert rl datasets to verl format")
parser.add_argument("--dataset_type", default="openr1-math", help="openr1-math or other")
parser.add_argument("--dataset_dir", default="data/OpenR1-Math-220k", help="path to the original dataset")
parser.add_argument("--output_dir", default="data/OpenR1-Math-220k/converted", help="where to put the converted dataset")
args = parser.parse_args()

if args.dataset_type == "openr1-math":
    dataset_part = "all"
    input_ds = load_dataset(args.dataset_dir, dataset_part, split="train")
    output_train_data = []
    output_test_data = []

    # conversion rules:
    # problem -> prompt (conversation style, array([{"role": "user", "content": problem}], dtype=object))
    # answer -> reward_model ({"ground_truth": answer (str), "style": "rule"})
    # uuid -> uuid
    for sample in input_ds:
        sample_prompt = np.array([{"role": "user", "content": sample["problem"]}])
        sample_rm = {"ground_truth": sample["answer"], "style": "rule"}
        sample_id = sample["uuid"]
        output_sample = {
            "data_source": "open-r1/OpenR1-Math-220k",
            "prompt": sample_prompt,
            "ability": "math",
            "reward_model": sample_rm,
            "extra_info": {
                "index": sample_id
            }
        }
        if np.random.uniform(0, 1) <= 0.99:
            output_train_data.append(output_sample)
        else:
            output_test_data.append(output_sample)

    output_train_df = pd.DataFrame(output_train_data)
    output_train_df.to_parquet(os.path.join(args.output_dir, f"{dataset_part}-train.parquet"))

    output_test_df = pd.DataFrame(output_test_data)
    output_test_df.to_parquet(os.path.join(args.output_dir, f"{dataset_part}-test.parquet"))
else:
    raise NotImplementedError()

