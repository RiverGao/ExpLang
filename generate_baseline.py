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
from tqdm import tqdm
import jsonlines
import argparse
from pathlib import Path
import logging
from math_verify import parse, verify
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
max_model_len = 8192
verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


def generate_responses(llm_instance, tokenizer_instance, batch, generation_params):
    prompts_to_sample = []
    
    for item in batch:
        assert len(item["messages"]) == 2

        item_messages = item["messages"][:-1]
        prompt = tokenizer_instance.apply_chat_template(
            item_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        if len(tokenizer_instance(prompt)["input_ids"]) > max_model_len // 2:
            continue  # remove overlong samples
        
        prompts_to_sample.append(prompt)
        
    responses = llm_instance.generate(prompts_to_sample, sampling_params=generation_params)
    
    results = []
    for item, response in zip(batch, responses):
        sample_id = item["id"]
        sample_input = item["messages"][0]
        sample_output = item["messages"][1]

        # assert len(response.outputs) == 1
        for output in response.outputs:
            assert output.text is not None
            generated_text =  output.text.strip()
            ground_truth_answer = item['ground_truth']  # could be None
            # verify the result's correctness
            gt_boxed = "\\boxed{" + ground_truth_answer + "}"
            score = 0.0
            try:
                score, _ = verify_func([gt_boxed], [generated_text])  # 0.0 or 1.0
            except Exception:
                pass
            is_correct = score > 0.5
            if not is_correct:
                # skip incorrect responses
                continue
            # is correct, save the result
            results.append({
                'id': sample_id,
                'messages': [
                    sample_input,
                    {"role": "assistant", "content": generated_text}
                ]
            })
            break  # only need one correct response per sample
        else:
            # no correct response found
            print(f"No correct response for sample id {sample_id}")
            results.append({
                'id': sample_id,
                'messages': item["messages"]
            })
    return results
    
    
def main():
    parser = argparse.ArgumentParser(description="Generate distillation data with language-specific prompts.")
    parser.add_argument('--multi_path', type=str, default='data/openr1-math_train.jsonl', help='Path to the multilingual data')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-32B', help='Path to the model directory.')
    parser.add_argument('--output_path', type=str, default='data/openr1-math_train_baseline.jsonl', help='Path to save generated data.')
    args = parser.parse_args()
    
    # load multilingual dataset
    queries = []
    with jsonlines.open(args.multi_path) as reader:
        for item in reader:
            queries.append(item)

    random_seed = 42
    
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
        n=5,
        temperature=0.1, 
        top_p=0.9, 
        max_tokens=max_model_len // 2,
    )
    
    # generate responses for each language
    logger.info(f"Generating responses in default mode")

    results = generate_responses(
        llm, 
        tokenizer,
        queries, 
        generation_params
    )

    with jsonlines.open(args.output_path, 'w') as writer:
        writer.write_all(results)
    
    
if __name__ == "__main__":
    main()

