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
import concurrent.futures
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
import numpy as np
import multiprocessing


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
max_model_len = 8192

# lang_prefixes = {
#     "en": "Okay",
#     "zh": "好的",
#     "es": "De acuerdo",
#     "fr": "D'accord",
#     "de": "In Ordnung",
#     "ja": "わかりました",
#     "ru": "Хорошо",
#     "it": "Va bene",
#     "pt": "Tudo bem",
#     "ko": "알겠습니다",
#     "ar": "حسنًا",
#     "th": "ตกลง",
#     "vi": "Được rồi"
# }

lang_prefixes = {
    "en": "Let's think about this problem in English.",
    "zh": "Let's think about this problem in Chinese.",
    "es": "Let's think about this problem in Spanish.",
    "fr": "Let's think about this problem in French.",
    "de": "Let's think about this problem in German.",
    "ja": "Let's think about this problem in Japanese.",
    "ru": "Let's think about this problem in Russian.",
    "it": "Let's think about this problem in Italian.",
    "pt": "Let's think about this problem in Portuguese.",
    "ko": "Let's think about this problem in Korean.",
    "ar": "Let's think about this problem in Arabic.",
    "th": "Let's think about this problem in Thai.",
    "vi": "Let's think about this problem in Vietnamese."
}

# lang_prefixes = {
#     "en": "Let's think about this problem in English.",
#     "zh": "让我们用中文思考这个问题。",
#     "es": "Pensemos en este problema en español.",
#     "fr": "Réfléchissons à ce problème en français.",
#     "de": "Lasst uns dieses Problem auf Deutsch betrachten.",
#     "ja": "この問題を日本語で考えてみましょう。",
#     "ru": "Давайте рассмотрим эту проблему на русском языке.",
#     "it": "Riflettiamo su questo problema in italiano.",
#     "pt": "Vamos pensar sobre este problema em português.",
#     "ko": "이 문제를 한국어로 생각해 봅시다.",
#     "ar": "دعونا نفكر في هذه المشكلة باللغة العربية.",
#     "th": "ลองมาพิจารณาปัญหานี้ในมุมมองของภาษาไทยกัน",
#     "vi": "Chúng ta hãy cùng suy nghĩ về vấn đề này bằng tiếng Việt."
# }

verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)

def generate_responses(args_tuple):
    lang, model_path, avail_gpu, batch, generation_params, lang_select = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = avail_gpu
    llm_instance = LLM(
        model=model_path, 
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        enable_prefix_caching=True
    )
    tokenizer_instance = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    prompts_to_sample = []
    if not lang_select:
        response_prefix = f"<think>\n{lang_prefixes[lang]}"
    else:
        response_prefix = f"<lang_select>\n{lang}\n</lang_select>\n<think>\n{lang_prefixes[lang]}"
    
    for item in batch:
        # print(item, type(item))
        # math-500 format:
        item_messages = [{"role": "user", "content": item["problem"]}]
        
        prompt = tokenizer_instance.apply_chat_template(
            item_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        prompt += "\n" + response_prefix

        if len(tokenizer_instance(prompt)["input_ids"]) > max_model_len // 2 or len(item_messages) != 1:
            continue
        prompts_to_sample.append(prompt)
        
    responses = llm_instance.generate(prompts_to_sample, sampling_params=generation_params)
    
    results = []
    for item, response in zip(batch, responses):
        generated_texts = [response_prefix + o.text.strip() for o in response.outputs]
        # math-500 format
        results.append({
            'id': item['unique_id'],
            'thinking_language': lang,
            'responses': generated_texts,
            'reference': item['answer']
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Infer evaluation data with language-specific prompts.")
    parser.add_argument('--dataset', type=str, default='math-500', help='math-500 or AIME')
    parser.add_argument('--model_path', type=str, default='./model/Qwen3-0.6B', help='Path to the model directory.')
    parser.add_argument('--output_dir', type=str, default='./infer_eval/Qwen3-0.6B', help='Directory to save generated data.')
    parser.add_argument('--lang_select', type=bool, default=False, help='Whether the model supports thinking language selection')
    parser.add_argument('--skip_generation', type=bool, default=False, help='Skip generation and use saved results')
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
    elif args.dataset == 'olymmath-easy':
        dataset = load_dataset("data/olymmath", "en-easy", split="test")
    elif args.dataset == 'olymmath-hard':
        dataset = load_dataset("data/olymmath", "en-hard", split="test")
    elif args.dataset == 'openr1-math':
        dataset = load_dataset("data/OpenR1-Math-220k", "all", split="train")
        dataset = dataset.rename_column("uuid", "unique_id")
        dataset = dataset.select(range(2000))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(Path(args.output_dir) / "analysis", exist_ok=True)
    os.makedirs(Path(args.output_dir) / "results", exist_ok=True)
    
    accs_per_sample = dict()

    # generate responses for each language
    for language in lang_prefixes.keys():
        logger.info(f"Processing language: {language}")

        if not args.lang_select:
            lang_results_output_path = Path(args.output_dir) / "analysis" / f"{args.dataset}_{language}_prefix.jsonl"
        else:
            lang_results_output_path = Path(args.output_dir) / "analysis" / f"{args.dataset}_{language}_select.jsonl"

        # Load or generate results
        if args.skip_generation and lang_results_output_path.exists():
            logger.info(f"Loading saved results from {lang_results_output_path}")
            lang_results = []
            with open(lang_results_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    lang_results.append(json.loads(line))
        else:
            # Load model and tokenizer
            avail_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            num_gpus = len(avail_gpus)
            
            generation_params = SamplingParams(
                n=5,
                temperature=0.1, 
                top_p=0.9, 
                max_tokens=max_model_len // 2,
            )
            
            # split dataset into num_gpus
            dataset_splits = [[] for _ in range(num_gpus)]
            for i, sample in enumerate(dataset):
                dataset_splits[i % num_gpus].append(sample)

            # Launch workers
            lang_results = []
            args_list = []
            for worker_id in range(num_gpus):
                args_list.append((
                    language, 
                    args.model_path, 
                    avail_gpus[worker_id],
                    dataset_splits[worker_id],
                    generation_params,
                    args.lang_select
                ))
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as ex:
                futures = [ex.submit(generate_responses, tup) for tup in args_list]
                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Workers"):
                    lang_results.extend(fut.result())
            
            print(f"Total generations collected: {len(lang_results)}")
            # save results to avoid re-generation
            with open(lang_results_output_path, 'w', encoding='utf-8') as f:
                for results_entry in lang_results:
                    f.write(json.dumps(results_entry, ensure_ascii=False) + '\n')
            logger.info(f"{language} results saved to {lang_results_output_path}")

        for item in lang_results:
            item_id = item['id']
            item_ref = item['reference']
            item_responses = item['responses']
            item_scores = []
            for response in item_responses:
                # verify response
                gt_boxed = "\\boxed{" + item_ref + "}"
                score = 0.0
                try:
                    score, _ = verify_func([gt_boxed], [response])  # 0.0 or 1.0
                except:
                    pass
                item_scores.append(score)
            item_acc = np.mean(item_scores)

            if item_id not in accs_per_sample:
                accs_per_sample[item_id] = {language: item_acc}
            else:
                if language not in accs_per_sample[item_id]:
                    accs_per_sample[item_id][language] = item_acc
                else:
                    print(f"ID {item_id} repeated, skip")
                    continue
    
    # filter samples with higher non-English acc than English
    interested_ids = []
    win_count = 0  # non-en better than en
    tie_count = 0  # non-en equal to en
    lose_count = 0  # non-en worse than en
    
    for id, accs in accs_per_sample.items():
        interested = False
        en_acc = accs["en"]
        for lang, acc in accs.items():
            if lang == "en":
                continue
            if acc > en_acc:
                interested = True
                win_count += 1
            elif acc == en_acc:
                tie_count += 1
            else:
                lose_count += 1
        if interested:
            interested_ids.append(id)
    
    total_comparisons = win_count + tie_count + lose_count
    win_rate = win_count / total_comparisons if total_comparisons > 0 else 0
    tie_rate = tie_count / total_comparisons if total_comparisons > 0 else 0
    lose_rate = lose_count / total_comparisons if total_comparisons > 0 else 0
    
    print(f"Filtered {len(interested_ids)} interested samples out of {len(accs_per_sample)} candidates")
    print(f"Non-EN vs. EN Win-Tie-Lose Rate: {win_rate:.4f}-{tie_rate:.4f}-{lose_rate:.4f}")
    print(f"Non-EN vs. EN Win-Tie-Lose Count: {win_count}-{tie_count}-{lose_count}")

    # save results for further evaluation
    if not args.lang_select:
        results_output_path = Path(args.output_dir) / "results" / f"interested_{args.dataset}_multilingual_prefix_mod1.txt"
    else:
        results_output_path = Path(args.output_dir) / "results" / f"interested_{args.dataset}_multilingual_select.txt"
    with open(results_output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(interested_ids) + "\n")
    logger.info(f"Interested IDs saved to {results_output_path}")

    
if __name__ == "__main__":
    main()
