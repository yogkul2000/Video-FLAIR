import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from lmms_eval.tasks._task_utils.flair import is_flair, flair_post, flair_answer

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
cache_name = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(base_cache_dir, cache_name)


def get_cache_dir(subject):
    if subject in ["Art", "Art_Theory", "Design", "Music"]:
        return "Art"
    elif subject in ["Biology", "Chemistry", "Geography", "Math", "Physics"]:
        return "Science"
    elif subject in ["History", "Literature", "Sociology", "Psychology"]:
        return "Humanities"
    elif subject in [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ]:
        return "Engineering"
    elif subject in [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ]:
        return "Medicine"
    elif subject in ["Accounting", "Economics", "Finance", "Manage", "Marketing"]:
        return "Business"
    else:
        raise ValueError(f"Subject {subject} not recognized.")


def videommmu_flair_kind(doc):
    answer = doc.get("answer", "")
    if isinstance(answer, list):
        answer_text = " ".join(str(x) for x in answer)
    else:
        answer_text = str(answer)
    question = str(doc.get("question", ""))
    if re.fullmatch(r"\s*-?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*,\s*-?(?:\d+(?:\.\d+)?|\.\d+))*\s*", answer_text):
        return "number"
    if any(x in answer_text + question for x in ["$", "\\", "=", "^", "_", "\\dfrac", "\\frac", "sqrt"]):
        return "math"
    return "short"


def videommmu_doc_to_visual(doc):
    subject = "_".join(doc["id"].split("_")[1:-1])


    videommmu_cache_dir = os.path.join(cache_dir, get_cache_dir(subject))

    video_path = doc["id"] + ".mp4"
    video_path = os.path.join(videommmu_cache_dir, video_path)

    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


def videommmu_doc_to_visual_interleave(doc):
    subject = "_".join(doc["id"].split("_")[1:-1])


    videommmu_cache_dir = os.path.join(cache_dir, get_cache_dir(subject))

    video_path = doc["id"] + ".mp4"
    video_path = os.path.join(videommmu_cache_dir, video_path)

    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    image = doc["image"]
    return [video_path, image]


def videommmu_doc_to_visual_question_only(doc):
    video_path = doc["id"] + "_image" + ".mp4"
    question_only_cache_dir = os.path.join(cache_dir, "question_only")
    video_path = os.path.join(question_only_cache_dir, video_path)

    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


def videommmu_doc_to_text_adaptation(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]


    question = question.replace("\\<image 1\\>", "<image 1>")
    question = question.replace("<Image 1>", "<image 1>")
    question = question.replace("<image 1>", "")


    if doc["question_type"] == "multiple-choice":
        parsed_options = parse_options(doc["options"])
        post_prompt = flair_post("mcq") if is_flair(lmms_eval_specific_kwargs) else lmms_eval_specific_kwargs["mcq_prompt"]
        return f"{question}\nOptions:\n{parsed_options}\n{post_prompt}"
    else:
        post_prompt = flair_post(videommmu_flair_kind(doc)) if is_flair(lmms_eval_specific_kwargs) else lmms_eval_specific_kwargs["open_ended_prompt"]
        return f"{question}\n{post_prompt}"


def videommmu_doc_to_text_adaptation_interleave(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]


    question = question.replace("\\<image 1\\>", "<image 1>")
    question = question.replace("<Image 1>", "<image 1>")


    if "<image 1>" not in question:
        question = question + " <image 1>"

    assert "<image 1>" in question


    if doc["question_type"] == "multiple-choice":
        parsed_options = parse_options(doc["options"])
        post_prompt = flair_post("mcq") if is_flair(lmms_eval_specific_kwargs) else lmms_eval_specific_kwargs["mcq_prompt"]
        return f"{question}\nOptions:\n{parsed_options}\n{post_prompt}"
    else:
        post_prompt = flair_post(videommmu_flair_kind(doc)) if is_flair(lmms_eval_specific_kwargs) else lmms_eval_specific_kwargs["open_ended_prompt"]
        return f"{question}\n{post_prompt}"


def videommmu_doc_to_text_no_preprompt(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    parsed_options = parse_options(doc["options"])
    question += "\n" + parsed_options

    return f"{question}"


def videommmu_doc_to_text_perception_comprehension(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    parsed_options = parse_options(doc["options"])
    post_prompt = flair_post("mcq") if is_flair(lmms_eval_specific_kwargs) else lmms_eval_specific_kwargs["mcq_prompt"]

    return f"{question}\nOptions:\n{parsed_options}\n{post_prompt}"


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]


    if all(option.startswith(f"{letter}.") for option, letter in zip(options, option_letters)):
        return "\n".join(options)


    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def videommmu_doc_to_answer(doc):
    return doc["answer"]


def videommmu_process_results(doc, results):
    raw_pred = results[0]
    pred = flair_answer(raw_pred)

    question_type = doc.get("question_type", "None")
    if question_type == "multiple-choice":
        index2ans, all_choices = get_multi_choice_info((doc["options"]))
        parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    else:
        parsed_pred = parse_open_response(pred)

    mmmu_acc = {
        "id": doc["id"],
        "subdomain": extract_subset_name(doc["id"]),
        "question_type": question_type,
        "answer": doc["answer"],
        "parsed_pred": parsed_pred,
        "raw_pred": raw_pred,
    }
    return {"mmmu_acc": mmmu_acc}


def extract_subset_name(input_string):
    split = input_string.split("_")[0]
    pattern = re.compile(rf"^{split}_(.+?)_\d+$")
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')


def videommmu_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)


    valid_results = [result for result in results if result["parsed_pred"] != "API Error"]

    for result in valid_results:
        subset_to_eval_samples[result["subdomain"]].append(result)

    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_videommmu(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict

    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results["num_example"] for cat_results in in_domain_cat_results.values()])
        printable_results["Overall-" + domain] = {
            "num": int(in_domain_data_num),
            "acc": round(in_domain_ins_acc, 5),
        }

        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
            }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    print(printable_results)
    return printable_results["Overall"]["acc"]


def calculate_ins_level_acc(results):
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


def eval_multi_choice(gold_i, pred_i):
    correct = False


    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i):
    correct = False
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def evaluate_videommmu(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        if sample["question_type"] == "multiple-choice":
            correct = eval_multi_choice(gold_i, pred_i)
        elif sample["question_type"] == "perception":
            correct = eval_multi_choice(gold_i, pred_i)
        else:
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if response == "API Error" or response == "":
        return "API Error"


    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "


    index_ans = True
    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []


    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append(choice)
            ans_with_period = True

    for choice in all_choices:
        if f"{choice}:" in response:
            candidates.append(choice)
            ans_with_colon = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)


    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False


    if len(candidates) == 0:
        pred_index = "No Answer Found."


    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_period:
                for can in candidates:
                    index = response.rfind(f"{can}.")
                    start_indexes.append(index)
            elif ans_with_colon:
                for can in candidates:
                    index = response.rfind(f"{can}:")
                    start_indexes.append(index)
            elif ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)

        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def extract_numbers(string):
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"

    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"

    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"


    numbers_with_commas = re.findall(pattern_commas, string)

    numbers_scientific = re.findall(pattern_scientific, string)

    numbers_simple = re.findall(pattern_simple, string)


    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def check_is_number(string):
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string):
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)

        string = round(string, 2)
        return [string]
    else:
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]
        return [string]


def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    if response == "API Error" or response == "":
        return "API Error"


    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [

            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
            "are ",
            "in total ",
            "total ",
            "identify ",
            "recognize ",
            "calculated as ",
            "counted as ",
            "measured as ",
            "observed as ",
            "concluded as ",
            "found to be ",
            "equals ",
            "determined to be ",
            "number of ",
            "value is ",
            "adds up to ",
            "have ",
            "has ",
        ]

        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = (
                None
            )
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()


            if shortest_key_response:
                if shortest_key_response.strip() not in [
                    ":",
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    ":",
                    "'",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list


    pred_list = list(set(pred_list))

    return pred_list


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices
