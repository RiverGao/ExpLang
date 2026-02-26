# merge the jsonl files (e.g. generated/Qwen3-4B/combined/*_100.jsonl) into a single parquet file
import os
import pandas as pd
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL files into a single Parquet file.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing JSONL files to merge.')
    parser.add_argument('--num_samples', type=int, help='Number of samples to process.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for Parquet files.')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    for jsonl_file in input_dir.glob(f"*_{args.num_samples}.jsonl"):
        print(f"Processing file: {jsonl_file}")
        df = pd.read_json(jsonl_file, lines=True)
        all_data.append(df)
    
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_file = output_dir / "train.parquet"
        
        merged_df.to_parquet(merged_file, index=False)
        print(f"Saved {len(merged_df)} samples to {merged_file}")
    else:
        print("No JSONL files found to merge.")

if __name__ == "__main__":
    main()

