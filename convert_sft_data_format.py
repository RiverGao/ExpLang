# convert tulu and lang_select datasets to sharegpt jsonl
import os
from datasets import load_dataset
import pandas as pd
import argparse
import jsonlines
from pathlib import Path


parser = argparse.ArgumentParser(description="convert tulu and lang_select datasets to sharegpt jsonl")
parser.add_argument("--dataset_type", default="tulu", help="tulu or lang_select")
parser.add_argument("--dataset_dir", default="data/tulu-3-sft-mixture", help="path to the original dataset")
parser.add_argument("--output_dir", default="data/tulu-3-sft-mixture/converted", help="where to put the converted dataset")
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
output_dir = Path(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# load the original dataset
if args.dataset_type == "tulu":
    ori_ds = load_dataset(args.dataset_dir, split="train")
    print(f"loaded {len(ori_ds)} data samples from {args.dataset_dir}")

    output_entries = []
    # convert dataset
    for item in ori_ds:
        output_entries.append(
            {
                "id": item["id"],
                "messages": item["messages"],
                "source": item["source"]
            }
        )

elif args.dataset_type == "lang_select":
    for subfolder in ["openr1-math"]:
        ori_ds = pd.read_parquet(dataset_dir / subfolder / "train.parquet")
        print(f"loaded {len(ori_ds)} data samples from {args.dataset_dir}/{subfolder}")
        output_entries = []

        for row in ori_ds.itertuples():
            messages = [
                {"role": "user", "content": row[3]},
                {"role": "assistant", "content": row[4]}
            ]
            output_entries.append(
                {
                    "id": row[1],
                    "messages": messages
                }
            )

        # write the entries
        with jsonlines.open(os.path.join(args.output_dir, f"{subfolder}_train.jsonl"), "w") as writer:
            writer.write_all(output_entries)