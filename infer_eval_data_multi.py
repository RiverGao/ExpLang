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

# lang_prefixes = {
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

# lang_prefixes = {
#     "en": "Let's think about this problem in English.",
#     "zh": "Let's think about this problem in Chinese.",
#     "es": "Let's think about this problem in Spanish.",
#     "fr": "Let's think about this problem in French.",
#     "de": "Let's think about this problem in German.",
#     "ja": "Let's think about this problem in Japanese.",
#     "ru": "Let's think about this problem in Russian.",
#     "it": "Let's think about this problem in Italian.",
#     "pt": "Let's think about this problem in Portuguese.",
#     "ko": "Let's think about this problem in Korean.",
#     "ar": "Let's think about this problem in Arabic.",
#     "th": "Let's think about this problem in Thai.",
#     "vi": "Let's think about this problem in Vietnamese."
# }

lang_prefixes = {
    "en": "Let's think about this problem in English.",
    "zh": "让我们用中文思考这个问题。",
    "es": "Pensemos en este problema en español.",
    "fr": "Réfléchissons à ce problème en français.",
    "de": "Lasst uns dieses Problem auf Deutsch betrachten.",
    "ja": "この問題を日本語で考えてみましょう。",
    "ru": "Давайте рассмотрим эту проблему на русском языке.",
    "it": "Riflettiamo su questo problema in italiano.",
    "pt": "Vamos pensar sobre este problema em português.",
    "ko": "이 문제를 한국어로 생각해 봅시다.",
    "ar": "دعونا نفكر في هذه المشكلة باللغة العربية.",
    "th": "ลองมาพิจารณาปัญหานี้ในมุมมองของภาษาไทยกัน",
    "vi": "Chúng ta hãy cùng suy nghĩ về vấn đề này bằng tiếng Việt."
}

def generate_responses(lang, llm_instance, tokenizer_instance, batch, generation_params, lang_select=False):
    prompts_to_sample = []
    if not lang_select:
        response_prefix = f"<think>\n{lang_prefixes[lang]}"
    else:
        response_prefix = f"<lang_select>\n{lang}\n</lang_select>\n<think>\n{lang_prefixes[lang]}"
    
    for item in batch:
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
        n=1,
        temperature=0.1, 
        top_p=0.9, 
        max_tokens=max_model_len // 2,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)

    results_dict = {}  # key: unique_id, value: {"reference": ref, "responses": [responses]}
    os.makedirs(Path(args.output_dir) / "results", exist_ok=True)
    
    # generate responses for each language
    for language in lang_prefixes.keys():
        logger.info(f"Generating responses for language: {language}")
        
        lang_results = generate_responses(
            language,
            llm, tokenizer,
            dataset, 
            generation_params,
            lang_select=args.lang_select
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
    if not args.lang_select:
        results_output_path = Path(args.output_dir) / "results" / f"{args.dataset}_multilingual_prefix_mod2.jsonl"
    else:
        results_output_path = Path(args.output_dir) / "results" / f"{args.dataset}_multilingual_select.jsonl"
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
