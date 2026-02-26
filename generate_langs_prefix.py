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
from langdetect import detect
import re
from math_verify import parse, verify
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
max_model_len = 8192

lang_adopt_rates = dict()
lang_adopt_rates = {
    "en": 0.18,
    "zh": 0.08,
    "es": 0.28,
    "fr": 0.21,
    "de": 0.21,
    "ja": 0.27,
    "ru": 0.20,
    "it": 0.23,
    "pt": 0.07,
    "ko": 0.39,
    "ar": 0.13,
    "th": 0.30,
    "vi": 0.30
}

lang_prefixes = {
    "en": "Okay",
    "zh": "好的",
    "es": "De acuerdo",
    "fr": "D'accord",
    "de": "In Ordnung",
    "ja": "わかりました",
    "ru": "Хорошо",
    "it": "Va bene",
    "pt": "Tudo bem",
    "ko": "알겠습니다",
    "ar": "حسنًا",
    "th": "ตกลง",
    "vi": "Được rồi"
}

thinking_pattern = re.compile(r"<think>([\s\S]*?)</think>")
verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)

def generate_responses(dataset_name, lang, llm_instance, tokenizer_instance, batch, generation_params, include_ground_truth=False):
    prompts_to_sample = []
    response_prefix = f"<think>\n{lang_prefixes[lang]}"
    
    for item in batch:
        if len(item["messages"]) > 2:
            continue  # remove multiturn samples

        item_messages = item["messages"][:-1]
        prompt = tokenizer_instance.apply_chat_template(
            item_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        prompt += "\n" + response_prefix

        if len(tokenizer_instance(prompt)["input_ids"]) > max_model_len // 2:
            continue  # remove overlong samples
        
        prompts_to_sample.append(prompt)
        
    responses = llm_instance.generate(prompts_to_sample, sampling_params=generation_params)
    
    queries = []
    results = []
    id_key_dict = {
        "tulu": "id",
        "openr1-math": "uuid"
    }
    for item, response in zip(batch, responses):
        queries.append({
            "id": item[id_key_dict[dataset_name]],
            "input": item["messages"][0]["content"]
        })

        generated_texts = [response_prefix + o.text.strip() for o in response.outputs]
        if not include_ground_truth:
            results.append({
                'id': item[id_key_dict[dataset_name]],
                'thinking_language': lang,
                'responses': generated_texts
            })
        else:
            if dataset_name != "openr1-math":
                raise NotImplementedError("Ground truth inclusion is only implemented for openr1-math dataset.")
            results.append({
                'id': item[id_key_dict[dataset_name]],
                'thinking_language': lang,
                'ground_truth': item["answer"],
                'responses': generated_texts
            })
            
    return queries, results
    
    
def main():
    parser = argparse.ArgumentParser(description="Generate distillation data with language-specific prompts.")
    parser.add_argument('--dataset', type=str, default='tulu', help='tulu or openr1-math')
    parser.add_argument('--model_path', type=str, default='./model/Qwen3-4B', help='Path to the model directory.')
    parser.add_argument('--output_dir', type=str, default='./generated/Qwen3-4B', help='Directory to save generated data.')
    parser.add_argument('--target_samples', type=int, default=1000, help='Target number of samples for each language.')
    parser.add_argument('--include_ground_truth', type=bool, default=False, help='Whether to include ground truth answers in the output for openr1-math dataset.')
    # parser.add_argument('--language', type=str, default='en', choices=list(lang_prefixes.keys()), help='Language for generation.')
    args = parser.parse_args()
    
    # load dataset
    if args.dataset == 'tulu':
        dataset = load_dataset("data/tulu-3-sft-mixture", split="train")
    elif args.dataset == 'openr1-math':
        dataset = load_dataset("data/OpenR1-Math-220k", split="train")
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

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
        n=1,
        temperature=0.1, 
        top_p=0.9, 
        max_tokens=max_model_len // 2,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)

    # generate responses for each language
    for language in lang_prefixes.keys():
        logger.info(f"Generating responses for language: {language}")

        # Limit to target_samples for demonstration purposes
        n_raw_samples = args.target_samples / lang_adopt_rates.get(language, 1.0) * 1.1
        data = dataset.shuffle(seed=random_seed).select(range(int(n_raw_samples)))
        random_seed += 1  # change seed for each language to get different shuffling

        for subfolder in ["queries", "results", "combined"]:
            os.makedirs(Path(args.output_dir) / subfolder, exist_ok=True)
        queries, results = generate_responses(
            args.dataset,
            language,
            llm, tokenizer,
            data, 
            generation_params,
            include_ground_truth=args.include_ground_truth
        )

        queries_output_path = Path(args.output_dir) / "queries" / f"{language}_{args.target_samples}.jsonl"
        with open(queries_output_path, 'w', encoding='utf-8') as f:
            for item in queries:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Queries saved to {queries_output_path}")

        results_output_path = Path(args.output_dir) / "results" / f"{language}_{args.target_samples}.jsonl"
        id2result = {}
        with open(results_output_path, 'w', encoding='utf-8') as f:
            for item in results:
                id2result[item['id']] = item
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Generated results saved to {results_output_path}")

        # combine queries and results (one query mapped to multiple responses in one result)
        combined_output_path = Path(args.output_dir) / "combined" / f"{language}_{args.target_samples}.jsonl"
        with open(combined_output_path, 'w', encoding='utf-8') as f:
            for query_item in queries:
                query_id = query_item['id']
                matching_result = id2result.get(query_id, None)
                result_thinking_language = matching_result['thinking_language'] if matching_result else "N/A"
                if matching_result:
                    for response in matching_result['responses']:
                        # detect the language of thinking part in the response
                        thinking_part_match = thinking_pattern.search(response)
                        if not thinking_part_match:
                            # skip this response
                            continue
                        else:
                            thinking_text = thinking_part_match.group(1).strip()
                            thinking_language = detect(thinking_text)
                            if result_thinking_language not in thinking_language:
                                # skip this response
                                continue
                        
                        if args.include_ground_truth and args.dataset == "openr1-math":
                            ground_truth_answer = matching_result['ground_truth']
                            # verify the response

                            # final answer correctness
                            gt_boxed = "\\boxed{" + ground_truth_answer + "}"
                            score = 0.0
                            try:
                                score, _ = verify_func([gt_boxed], [response])  # 0.0 or 1.0
                            except:
                                pass
                            is_correct = score > 0.5
                            if not is_correct:
                                # skip incorrect responses
                                continue
                            
                        # modify response with <lang_select> and </lang_select> tags
                        response = f"<lang_select>\n{result_thinking_language}\n</lang_select>\n" + response
                        combined_item = {
                            'id': query_id,
                            'thinking_language': result_thinking_language,
                            'input': query_item['input'],
                            'response': response,
                            'ground_truth': ground_truth_answer if args.include_ground_truth and args.dataset == "openr1-math" else None
                        }
                        f.write(json.dumps(combined_item, ensure_ascii=False) + '\n')
    
    
if __name__ == "__main__":
    main()

