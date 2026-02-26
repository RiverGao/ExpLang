# generate model responses to common instruction datasets with language-specific system prompts
# language pool: ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ru', 'it', 'pt', 'ko']
# language-specific system prompts suffix: "You must think and answer questions in {language}."
# generation framework: vllm
# model: ./model/Qwen3-4B
# data: allenai/tulu-3-sft-mixture

import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
max_model_len = 8192

def generate_responses(llm_instance, tokenizer_instance, batch, generation_params):
    prompts_to_sample = []
    
    for item in batch:
        # math-500 format:
        item_messages = [{"role": "user", "content": item["problem"]}]
        
        prompt = tokenizer_instance.apply_chat_template(
            item_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        if len(tokenizer_instance(prompt)["input_ids"]) > max_model_len // 2 or len(item_messages) != 1:
            continue
        prompts_to_sample.append(prompt)
        
    responses = llm_instance.generate(prompts_to_sample, sampling_params=generation_params)
    
    results = []
    for item, response in zip(batch, responses):
        generated_texts = [o.text.strip() for o in response.outputs]
        # math-500 format
        results.append({
            'id': item['unique_id'],
            'thinking_language': "self",
            'responses': generated_texts,
            'reference': item['answer']
        })
    return results
    
    
def main():
    parser = argparse.ArgumentParser(description="Infer evaluation data with language-specific prompts.")
    parser.add_argument('--dataset', type=str, default='math-500', help='math-500 or AIME')
    parser.add_argument('--model_path', type=str, default='./model/Qwen3-0.6B', help='Path to the model directory.')
    parser.add_argument('--output_dir', type=str, default='./infer_eval/Qwen3-0.6B', help='Directory to save generated data.')
    args = parser.parse_args()
    
    # load dataset
    if args.dataset == 'math-500':
        dataset = load_dataset("data/math-500", split="test")
    elif args.dataset == 'aime24':
        dataset = load_dataset("data/aime_2024", split="train")
        dataset = dataset.rename_column("id", "unique_id")
    elif args.dataset == 'aime25':
        dataset = load_dataset("data/aime_2025", split="train")
        dataset = dataset.rename_column("id", "unique_id")
    elif args.dataset =='olymmath-easy':
        dataset = load_dataset("data/olymmath", "en-easy", split="test")
    elif args.dataset =='olymmath-hard':
        dataset = load_dataset("data/olymmath", "en-hard", split="test")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # load model and tokenizer
    num_gpus = torch.cuda.device_count()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    llm = LLM(
        model=args.model_path, 
        tensor_parallel_size=num_gpus,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        enable_prefix_caching=True
    )
    
    generation_params = SamplingParams(
        n=12,
        temperature=0.1, 
        top_p=0.9, 
        max_tokens=max_model_len // 2,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)

    results_dict = {}  # key: unique_id, value: {"reference": ref, "responses": [responses]}
    os.makedirs(Path(args.output_dir) / "results", exist_ok=True)
    
    # generate multiple responses for each self-choice run
    logger.info(f"Generating responses by self-choice")
    
    lang_results = generate_responses(
        llm, tokenizer,
        dataset, 
        generation_params,
    )

    for item in lang_results:
        item_id = item['id']
        item_ref = item['reference']
        item_responses = item['responses']
        
        if item_id not in results_dict:
            results_dict[item_id] = {"reference": item_ref, "responses": item_responses}
        else:
            assert results_dict[item_id]["reference"] == item_ref
            results_dict[item_id]["responses"].extend(item_responses)

    # save results for further evaluation
    results_output_path = Path(args.output_dir) / "results" / f"{args.dataset}_self.jsonl"
    with open(results_output_path, 'w', encoding='utf-8') as f:
        for unique_id, record in results_dict.items():
            results_entry = {
                "id": unique_id,
                "reference": record["reference"],
                "responses": record["responses"]
            }
            f.write(json.dumps(results_entry, ensure_ascii=False) + '\n')
    logger.info(f"Inference results saved to {results_output_path}")

    
if __name__ == "__main__":
    main()
