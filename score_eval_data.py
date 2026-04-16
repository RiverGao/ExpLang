import re
from math_verify import parse, verify
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
import argparse
import jsonlines
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer


# prefixes_to_langs = {
#     "Okay": "en",
#     "好的": "zh",
#     "De acuerdo": "es",
#     "D'accord": "fr",
#     "In Ordnung": "de",
#     "わかりました": "ja",
#     "Хорошо": "ru",
#     "Va bene": "it",
#     "Tudo bem": "pt",
#     "알겠습니다": "ko",
#     "حسنًا": "ar",
#     "ตกลง": "th",
#     "Được rồi": "vi",
#     "Oke": "id",
#     "בסדר": "he",
#     "În regulă": "ro",
#     "Sawa": "sw"
# }

# prefixes_to_langs = {
#     "Let's think about this problem in English.": "en",
#     "Let's think about this problem in Chinese.": "zh",
#     "Let's think about this problem in Spanish.": "es",
#     "Let's think about this problem in French.": "fr",
#     "Let's think about this problem in German.": "de",
#     "Let's think about this problem in Japanese.": "ja",
#     "Let's think about this problem in Russian.": "ru",
#     "Let's think about this problem in Italian.": "it",
#     "Let's think about this problem in Portuguese.": "pt",
#     "Let's think about this problem in Korean.": "ko",
#     "Let's think about this problem in Arabic.": "ar",
#     "Let's think about this problem in Thai.": "th",
#     "Let's think about this problem in Vietnamese.": "vi"
# }

prefixes_to_langs = {
    "Let's think about this problem in English.": "en",
    "让我们用中文思考这个问题。": "zh",
    "Pensemos en este problema en español.": "es",
    "Réfléchissons à ce problème en français.": "fr",
    "Lasst uns dieses Problem auf Deutsch betrachten.": "de",
    "この問題を日本語で考えてみましょう。": "ja",
    "Давайте рассмотрим эту проблему на русском языке.": "ru",
    "Riflettiamo su questo problema in italiano.": "it",
    "Vamos pensar sobre este problema em português.": "pt",
    "이 문제를 한국어로 생각해 봅시다.": "ko",
    "دعونا نفكر في هذه المشكلة باللغة العربية.": "ar",
    "ลองมาพิจารณาปัญหานี้ในมุมมองของภาษาไทยกัน": "th",
    "Chúng ta hãy cùng suy nghĩ về vấn đề này bằng tiếng Việt.": "vi",
    "Mari kita pikirkan masalah ini dalam bahasa Indonesia.": "id",
    "בואו נחשוב על הבעיה הזו בעברית.": "he",
    "Să ne gândim la această problemă în limba românească.": "ro",
    "Hebu tufikirie tatizo hili kwa Kiswahili.": "sw"   
}

selection_pattern = re.compile(r"<lang_select>\n(.*?)\n</lang_select>")
thinking_pattern = re.compile(r"<think>([\s\S]*?)</think>")
languages = [
    Language.ENGLISH, 
    Language.CHINESE, 
    Language.SPANISH, 
    Language.FRENCH, 
    Language.GERMAN, 
    Language.JAPANESE, 
    Language.RUSSIAN, 
    Language.ITALIAN, 
    Language.PORTUGUESE, 
    Language.KOREAN, 
    Language.ARABIC, 
    Language.THAI, 
    Language.VIETNAMESE,
    Language.INDONESIAN,
    Language.HEBREW,
    Language.ROMANIAN,
    Language.SWAHILI
]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)

# def extract(output, use_cot=True):
#     if use_cot:
#         if 'Final Answer' in output:
#             output_tail = output.split('Final Answer')[-1]
#         elif '\n\n' in output:
#             output_tail = output.split('\n\n')[-1]
#         else:
#             output_tail = output[-32:]
#     else:
#         output_tail = output
    
#     # extract the final answer using regex, pattern: "$$\n\\boxed{<answer>}\n$$"
#     pattern = re.compile(r"\$\$\s*\\boxed\{([\s\S]*?)\}\s*\$\$")
#     match = pattern.search(output_tail)
#     if match:
#         answer = match.group(1).strip()
#         answer = answer.replace('\\dfrac', '\\frac')
#         return answer
#     else:
        # return None
    
def evaluate_response(ground_truth_answer, output_response):
    extracted_answer = output_response
    if extracted_answer is None:
        return False
    gold = "\\boxed{" + ground_truth_answer + "}"
    score = 0.0
    try:
        score, _ = verify_func([gold], [extracted_answer])  # 0.0 or 1.0
    except:
        pass
    is_correct = score > 0.5
    return is_correct

def evaluate_one_sample(tokenizer_instance, sample, language, control_method, filter_outputs=False, strict_compliant=False):
    ground_truth_answer = sample['reference']
    outputs = sample['responses']  # list of model outputs
    correctness_results = dict()
    language_results = []  # whether it follows language selection
    length_results = []
    if filter_outputs:
        filtered_outputs = []  # store compliant and correct outputs
    # assert len(outputs) == len(prefixes_to_langs) - 1  # excluding 'en'

    for output in outputs:
        # check language compliance
        if language == 'en':
            expected_lang = 'en'
        else:
            if control_method == 'prefix':
                for prefix, lang in prefixes_to_langs.items():
                    if output.startswith(f"<think>\n{prefix}"):
                        expected_lang = lang
                        break
                else:
                    if language == "self":
                        expected_lang = "unknown"
                    else:
                        raise ValueError("Output does not start with any known language prefix.")
            elif control_method == 'lang_select':
                # extract the language tag with regex
                match = selection_pattern.search(output)
                lang_tag = match.group(1) if match else None
                if lang_tag:
                    expected_lang = lang_tag
                else:
                    raise ValueError("Language tag not found in output.")
            else:
                raise ValueError("Unknown control method.")
        
        thinking_part_match = thinking_pattern.search(output)
        if not thinking_part_match:
            # check if thinking part exists but </think> is missing
            if "<think>" in output:
                thinking_text = output.split("<think>")[-1]
            else:
                raise ValueError(f"Thinking part not found in output: {output}")
        else:
            # thinking_text = thinking_part_match.group(0).strip("<think>").strip("</think>").strip()
            thinking_text = thinking_part_match.group(1).strip()
        thinking_language = detector.detect_language_of(thinking_text)
        if thinking_language:
            thinking_language = thinking_language.iso_code_639_1.name.lower()
        is_compliant = expected_lang == thinking_language
        language_results.append(is_compliant)

        is_correct = evaluate_response(ground_truth_answer, output)
        if strict_compliant and not is_compliant:
            is_correct = False  # if not compliant, mark as incorrect
        if expected_lang not in correctness_results:
            correctness_results[expected_lang] = []
        correctness_results[expected_lang].append(is_correct)
        
        thinking_len = len(tokenizer_instance(thinking_text, add_special_tokens=False)["input_ids"])
        # if thinking_len > 4096:
        #     breakpoint()
        # thinking_len = len(thinking_text.strip())  # count character numbers
        length_results.append(thinking_len)

        if filter_outputs:
            if is_compliant and is_correct:
                filtered_outputs.append({"language": expected_lang, "output": output})
    
    if not filter_outputs:
        return correctness_results, language_results, length_results
    else:
        return correctness_results, language_results, length_results, filtered_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model responses against ground truth answers and give accuracy and pass@k.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input jsonl file containing samples.')
    parser.add_argument('--language', type=str, default='en', help='en or multilingual or self')
    parser.add_argument('--control_method', type=str, default='prefiex', help='prefix or lang_select')
    parser.add_argument('--model_path', type=str, default='model/Qwen3-4B', help='path of the model (to get tokenizer)')
    parser.add_argument('--filter_outputs', action='store_true', help='Whether to filter outputs to only compliant and correct ones.')
    parser.add_argument('--compliant_only', action='store_true', help='Whether to evaluate only on compliant outputs.')
    parser.add_argument('--strict_compliant', action='store_true', help='If set, mark non-compliant outputs as incorrect.')
    args = parser.parse_args()

    # Load input samples
    samples = []
    with jsonlines.open(args.input_file, 'r') as reader:
        for obj in reader:
            samples.append(obj)
    print(f"Loaded {len(samples)} samples from {args.input_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    # Evaluate each sample
    all_acc = []
    all_passk = [] # k = all
    all_compliance = []
    all_lengths = []  # thinking length
    per_language_acc = dict()
    
    for sample in samples:
        if args.filter_outputs:
            correctness_results, language_results, length_results, filtered_outputs = evaluate_one_sample(
                tokenizer,
                sample, 
                language=args.language, 
                control_method=args.control_method, 
                filter_outputs=True
            )
            # breakpoint()
            with open(f"filtered_outputs.txt", 'a', encoding='utf-8') as f:
                if filtered_outputs:
                    for entry in filtered_outputs:
                        f.write(f"{entry}\n")
                    f.write("\n\n")
                else:
                    f.write("No compliant and correct outputs.\n\n")
                    
        else:
            correctness_results, language_results, length_results = evaluate_one_sample(
                tokenizer,
                sample, 
                language=args.language, 
                control_method=args.control_method,
                strict_compliant=args.strict_compliant
            )

        flat_correctness_results = []  # all results across languages for one sample
        for lang, results in correctness_results.items():
            if lang not in per_language_acc:
                per_language_acc[lang] = []
            per_language_acc[lang].extend(results)  # accumulate language-wise results across samples
            flat_correctness_results.extend(results)
        
        if args.compliant_only:
            # exclude samples that are not compliant, for forced generation
            processed_flat_correctness_results = []
            for cres, lres in zip(flat_correctness_results, language_results):
                if lres:
                    processed_flat_correctness_results.append(cres)
            flat_correctness_results = processed_flat_correctness_results
        
        acc = sum(flat_correctness_results) / len(flat_correctness_results) if len(flat_correctness_results) > 0 else 0.0  # accuracy for this sample
        all_acc.append(acc)

        passk = 1.0 if any(flat_correctness_results) else 0.0  # pass@k for this sample, k = all
        all_passk.append(passk)

        compliance_rate = sum(language_results) / len(language_results)  # compliance rate for this sample
        all_compliance.append(compliance_rate)

        mean_length = sum(length_results) / len(length_results)  # mean thinking length for this sample 
        all_lengths.append(mean_length)
        

    # report to command line
    overall_acc = sum(all_acc) / len(all_acc)
    overall_passk = sum(all_passk) / len(all_passk)
    overall_compliance = sum(all_compliance) / len(all_compliance)
    overall_length = sum(all_lengths) / len(all_lengths)
    print(f"Overall Accuracy: {100 * overall_acc:.1f}")
    print(f"Overall Pass@k: {100 * overall_passk:.1f}")
    print(f"Overall Thinking Length: {overall_length:.1f}")
    print(f"Overall Language Compliance: {100 * overall_compliance:.1f}")

    # for lang in prefixes_to_langs.values():
    #     if lang in per_language_acc:
    #         results = per_language_acc[lang]
    #         lang_acc = sum(results) / len(results)
    #         lang_chosen_rate = len(results) / (len(samples) * 12)
    #     else:
    #         lang_acc = 0.0
    #         lang_chosen_rate = 0.0
    #     print(f"Lang {lang} Acc: {100 * lang_acc:.1f};\tRate: {lang_chosen_rate * 100:.1f}%")

