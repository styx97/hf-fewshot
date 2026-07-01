from tqdm.auto import tqdm
import json
from pathlib import Path
import numpy as np
import re

import argparse

import torch
from hf_fewshot.models import (
    # MistralFewShot,
    HFFewShot,
    LlamaFewShot,
    GPTFewShot,
    Gemma2FewShot,
    Gwen2FewShot,
    Qwen3FewShot,
    Gemma3FewShot,
    display_gpu_status,
    get_unused_gpu_memory,
    get_logsoftmax
)

from hf_fewshot.prompting_utils import (
    prep_prompt,
    resolve_text_key,
    load_yaml,
    load_jsonlines,
    load_records,
    load_json,
    write_jsonlines,
    read_md
)

model_map = {
    "hf-general": HFFewShot,
    "llama": LlamaFewShot,
    "gpt": GPTFewShot,
    "qwen2": Gwen2FewShot,
    "qwen3": Qwen3FewShot,
    "gemma2": Gemma2FewShot,
    "gemma3": Gemma3FewShot,
    # "mistral": MistralFewShot,
}


def get_weighted_numeric_score(preference: dict[str, float] | None) -> float | None:
    """
    Compute a weighted numeric score from label probabilities.

    This expects labels that can be cast to floats (e.g., "1"-"5").
    Returns None when labels are non-numeric or probabilities are invalid.
    """
    if not preference:
        return None

    weighted_sum = 0.0
    total_prob = 0.0

    for label, prob in preference.items():
        try:
            label_value = float(label)
            prob_value = float(prob)
        except (TypeError, ValueError):
            # Non-numeric labels are not supported for weighted numeric score.
            return None

        if not np.isfinite(prob_value) or prob_value < 0:
            continue

        weighted_sum += label_value * prob_value
        total_prob += prob_value

    if total_prob <= 0:
        return None

    return weighted_sum / total_prob


def normalize_label_logprobs(label_logprobs: dict[str, float]) -> dict[str, float]:
    """
    Convert per-label logprobs into probabilities normalized over the provided labels.

    Labels with non-finite logprobs receive probability 0.
    """
    if not label_logprobs:
        return {}

    labels = list(label_logprobs.keys())
    logprob_values = np.array([label_logprobs[label] for label in labels], dtype=float)
    finite_mask = np.isfinite(logprob_values)

    if not finite_mask.any():
        return {label: 0.0 for label in labels}

    normalized = np.zeros_like(logprob_values, dtype=float)
    finite_values = logprob_values[finite_mask]
    max_logprob = np.max(finite_values)
    exp_values = np.exp(finite_values - max_logprob)
    denom = exp_values.sum()

    if denom > 0:
        normalized[finite_mask] = exp_values / denom

    return {label: float(prob) for label, prob in zip(labels, normalized)}

def get_option_preferences(model: LlamaFewShot,
                           logprobs: np.array,
                           options: list[str]) -> np.array:
    """
    Given the logprobs of the options, find the model preferences for each option.

    The answer is read at the FIRST non-whitespace generated token -- the position where
    the model commits to its answer. Leading whitespace/newline tokens (common with chat
    templates, e.g. Qwen3 emits " 5" = [220, 16]) are skipped. We deliberately do NOT
    scan downstream for a stray label token: if the first content token is not one of the
    options -- the model spelled out "five", wrapped it in prose, leaked a <think> tag,
    etc. -- it did not follow the prompt, and we return None for that example rather than
    fabricating a score from a digit buried later in the output. None surfaces as a row
    with no preferences/weighted score, so non-compliant generations stay visible.

    Assumes greedy decoding (do_sample=False): the generated token at each position is the
    argmax of that position's distribution.
    """
    # NOTE: single-token options only; multi-token labels would need per-token positions.
    option_token_dict = {
        option: model.tokenizer.encode(option, add_special_tokens=False)
        for option in options
    }
    label_token_ids = {ids[0] for ids in option_token_dict.values()}

    preferences = []
    num_examples, num_positions = logprobs.shape[0], logprobs.shape[1]

    for index in range(num_examples):
        # Walk to the first non-whitespace generated (argmax) token: the answer position.
        answer_pos = None
        for pos in range(num_positions):
            gen_token = int(np.argmax(logprobs[index, pos]))
            if model.tokenizer.decode([gen_token]).strip():
                answer_pos = pos
                break

        # Non-compliant: only whitespace, or the committed token isn't one of the options.
        if answer_pos is None or int(np.argmax(logprobs[index, answer_pos])) not in label_token_ids:
            preferences.append(None)
            continue

        option_logprobs = {
            option: float(logprobs[index, answer_pos, option_token_dict[option]].sum())
            for option in options
        }
        # Normalize over options so preferences are directly interpretable.
        option_probs = normalize_label_logprobs(option_logprobs)

        preferences.append(option_probs)


    return preferences


def get_logprobs(scores):
    """
    This function takes raw logit scores and returns logprobs for each output label

    Note: Changes shape from (max_new_tokens, batch_size, vocab_size) -> (batch_size, max_new_tokens, vocab_size)
    """

    # find out the batch size
    batch_size = scores[0].shape[0]
    logprobs = []
    for index in range(batch_size):
        curr_logprobs = []
        for token_output in scores:
            token_output_logprobs = get_logsoftmax(token_output[index])
            curr_logprobs.append(token_output_logprobs[0].detach().cpu().numpy())

        logprobs.append(curr_logprobs)

    return np.array(logprobs)


def parse_labels_config(labels_config: str | list[str] | dict) -> list[str]:
    """
    Parse the labels config and return a list of labels

    Examples
    --------
    >>> parse_labels_config("0-10")
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    >>> parse_labels_config("1,2")
    ['1', '2']
    >>> parse_labels_config("1,2,3")
    ['1', '2', '3']
    >>> parse_labels_config("-5 - 5")
    ['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5']
    """
    if isinstance(labels_config, list):
        try:
            labels = list(map(str, labels_config))
        except:
            raise ValueError("Labels should be a list of strings or integers")
    elif isinstance(labels_config, str):
        if ',' in labels_config:
            labels = [l.strip() for l in labels_config.split(',')]
        elif re.search(r'^[+-]?\d+\s?-\s?[+-]?\d+$', labels_config):
            low_high = re.split(r'(?<=\d)\s?-\s?(?=[+-]?\d)', labels_config)
            assert len(low_high) == 2, "Labels should be a string in the format '(±)<low> - (±)<high>'"
            try:
                low_high = list(map(int, low_high))
            except:
                raise ValueError("Labels should be a string in the format '(±)<low> - (±)<high>'")
            labels = list(map(str, list(range(low_high[0], low_high[1]+1))))
        else:
            ValueError("if labels are specified as string, they should use comma separated list (for pairwise) or integer range in format '(±)<low> - (±)<high>' (for pointwise)")
    elif isinstance(labels_config, dict):
        assert "low" in labels_config and "high" in labels_config, "Labels should be a dict with keys 'low' and 'high'"
        assert isinstance(labels_config["low"], (int, str)) and isinstance(labels_config["high"], (int, str)), "Labels should be a dict with keys 'low' and 'high' as integers"
        if isinstance(labels_config["low"], str):
            try:
                labels_config["low"] = int(labels_config["low"])
            except:
                raise ValueError("Labels should be a dict with keys 'low' and 'high' as integers")
        if isinstance(labels_config["high"], str):
            try:
                labels_config["high"] = int(labels_config["high"])
            except:
                raise ValueError("Labels should be a dict with keys 'low' and 'high' as integers")
        assert labels_config["low"] < labels_config["high"], "For scoring, Label 'low' should be a smaller integer than label 'high'"
        labels = list(map(str, list(range(labels_config["low"], labels_config["high"] + 1))))
    else:
        raise ValueError("Labels should be a list of strings or a string in the format 'low-high'")

    return labels


def load_prompts_and_exemplars(config: dict) -> tuple[str, list[dict]]:
    """
    Load the prompt and exemplars from the config file
    If no exemplars are provided, return None

    Accepts prompt input formats:
    1. A json file that contains the prompt with the keys "zero_shot" and "followup"
    2. A markdown directory that contains the prompt in the files "zero_shot.md" and "followup.md"
    3. A strict markdown folder mode (input_type=md_folder) that always requires both files
    """
    prompt_details = config["prompt_details"]
    prompt_path = Path(prompt_details["path"])

    if prompt_details['input_type'] == "json":
        prompts = load_json(prompt_path)
        prompt = prompts[config["prompt_details"]["prompt_name"]]

    elif prompt_details['input_type'] == "md":
        # in this case, read the md files and return the content
        # filepath/zero_shot.md and filepath/followup.md
        zero_shot_filepath = prompt_path / "zero_shot.md"
        # first, verify that the files exist
        assert zero_shot_filepath.is_file(), f"Zero-shot prompt file not found at {zero_shot_filepath}"
        zero_shot = read_md(zero_shot_filepath)

        if config["exemplars"]["use_exemplars"]:
            followup_filepath = prompt_path / "followup.md"
            assert followup_filepath.is_file(), f"Follow-up prompt file not found at {followup_filepath}"
            followup = read_md(followup_filepath)
        else:
            followup = None

        prompt = {"zero_shot": zero_shot,
                 "followup": followup}

    elif prompt_details['input_type'] == "md_folder":
        # strict markdown-folder mode: both files must exist
        zero_shot_filepath = prompt_path / "zero_shot.md"
        followup_filepath = prompt_path / "followup.md"

        assert zero_shot_filepath.is_file(), f"Zero-shot prompt file not found at {zero_shot_filepath}"
        assert followup_filepath.is_file(), f"Follow-up prompt file not found at {followup_filepath}"

        prompt = {
            "zero_shot": read_md(zero_shot_filepath),
            "followup": read_md(followup_filepath)
        }

    else:
        raise ValueError("prompt_details.input_type must be one of: json, md, md_folder")

    if config['exemplars']['use_exemplars']:
        exemplars_path = config["exemplars"]["path"]
        exemplars = load_records(exemplars_path, config["exemplars"].get("format")) if exemplars_path != 'None' else None
        if config["exemplars"]["shuffle"]:
            np.random.seed(config["exemplars"]["seed"])
            np.random.shuffle(exemplars)
        num_exemplars = config["exemplars"]["num_exemplars"]
        return prompt, exemplars[:num_exemplars]

    return prompt, None


def prepare_output_file(config: dict) -> Path:
    """
    Create and return the output file path. If the file already exists, return the path to the existing file
    """

    if "output_file" in config["output"]:
        outfile = Path(config["output"]["output_dir"]) / config["output"]["output_file"]
    else:
        outfile = Path(config['output']['output_dir']) / f"{config['output']['task_name']}_on_{config['dataset']['dataset_name']}_{config['model_details']['model_name']}.jsonl"

    if not outfile.is_file():
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.touch()

    return outfile


def prepare_initial_data(config: dict,
                        outfile: Path,
                        exemplars: list[dict],
                        prompt: str) -> tuple[list[str], list[dict], str]:
    """
    Prepare the initial data for the model by generating the
    prompts and filtering out the data that has already been processed
    """

    dataset = load_records(config["dataset"]["path"], config["dataset"].get("format"))
    id_key = config["dataset"]["id_key"]
    text_key = config["dataset"].get("text_key")
    input_vars = config["prompt_details"]["input_vars"]
    output_var = config["prompt_details"]["output_var"]

    assert id_key in dataset[0].keys(), f"ID key {id_key} not found in dataset"
    if text_key:
        assert text_key in dataset[0].keys(), \
            f"text_key {text_key!r} not found in dataset: {set(dataset[0].keys())}"

    # Validate prompt variables against dataset keys, accounting for the text_key
    # alias (text_key is exposed to prompts as {text}).
    dataset_vars = set(resolve_text_key(dataset[0], text_key).keys())
    assert set(input_vars).issubset(dataset_vars), \
        (f"Variables in the prompt: {input_vars} are not found in the dataset: {dataset_vars}")

    if exemplars:
        all_vars = set(input_vars + [output_var])
        exemplar_vars = set(resolve_text_key(exemplars[0], text_key).keys())
        assert set(all_vars).issubset(exemplar_vars), \
            (f"Variables in the prompt: {all_vars} are not the same as the variables in the exemplar: {exemplar_vars}")

    existing_data = load_jsonlines(outfile)
    if existing_data:
        existing_ids = {d[id_key] for d in existing_data}
        dataset = [d for d in dataset if d[id_key] not in existing_ids]
        print(f"Found {len(existing_data)} data items in the output file.")
        print(f"Running inference on {len(dataset)} items.")

    query_texts = [prep_prompt(d, output_var, prompt=prompt, exemplars=exemplars, text_key=text_key) for d in dataset]

    return query_texts, dataset, id_key


def run_inference(model,
                  query_texts,
                  batch_size,
                  outfile,
                  id_values,
                  id_key,
                  api_model,
                  dynamic_batching
            ):
    has_labels = hasattr(model, "label_id_map") and model.label_id_map
    if not has_labels:
        print("Model does not have labels. Running inference without obtaining preferences")

    print("Starting inference loop")
    pbar = tqdm(total=len(query_texts), desc='Running Inference')

    print("Writing responses to: ", outfile)

    n_scored = 0       # rows where the model committed to a label at the first content token
    n_noncompliant = 0  # labeled rows where it did not (None preference)

    with open(outfile, "a+") as f:
        i = 0
        while i < len(query_texts):
            try:
                batch_query_texts = query_texts[i:i + batch_size]
                ids = id_values[i:i + batch_size]
                batched_output = model.generate_answer_batch_logprobs(batch_query_texts)
                if api_model:
                    preferences = [
                        normalize_label_logprobs(
                            {lab: logprobs.get(lab, -np.inf) for lab in model.label_id_map.keys()}
                        ) if has_labels else None
                        for logprobs in batched_output["scores"]
                    ]
                else:
                    logprobs = get_logprobs(batched_output["scores"])
                    preferences = get_option_preferences(model, logprobs, list(model.label_id_map.keys())) if has_labels else [None] * len(batch_query_texts)

                #import ipdb; ipdb.set_trace()

                for item_id, preference, answer in zip(ids, preferences, batched_output["answers"]):
                    output = {
                        id_key: item_id,
                        "output": answer,
                    }
                    if preference:
                        output["preferences"] = preference
                        weighted_score = get_weighted_numeric_score(preference)
                        if weighted_score is not None:
                            output["weighted_likert_score"] = round(weighted_score, 4)
                        n_scored += 1
                    elif has_labels:
                        n_noncompliant += 1
                    f.write(json.dumps(output) + "\n")

                i += batch_size
                pbar.update(batch_size)

                if api_model:
                    continue

                unused_gpu_mem = get_unused_gpu_memory()

                # if dynamic batching is turned on, increase batch size by 2
                if unused_gpu_mem > 40 and dynamic_batching:
                    print("Unused GPU memory: ", unused_gpu_mem)
                    batch_size += 2
                    print(f"Increasing batch size to {batch_size}")
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as e:
                print("Out of memory error. Reducing batch size")
                print(e)
                display_gpu_status()
                if batch_size == 1:
                    print("Batch size of 1 too large for GPU. Aborting")

                batch_size = max(1, batch_size - 2)
                print(f"Reducing batch size to {batch_size}")
                torch.cuda.empty_cache()

            except Exception as e:
                print("An error occurred:")
                print(e)
                # NOTE: when an error arises in a batch, the while loop would continues infinitely so we break here
                break

            if not api_model:
                torch.cuda.empty_cache()

    # report the length of the file
    with open(outfile, "r") as f:
        lines = f.readlines()
        print(f"Total lines written to {outfile}: {len(lines)}")

    if has_labels:
        total = n_scored + n_noncompliant
        if total:
            rate = n_noncompliant / total
            print(f"Non-compliant (no label at first content token): "
                  f"{n_noncompliant}/{total} ({rate:.0%})")
            if rate > 0.1:
                print("WARNING: high non-compliance -- check reasoning-mode <think> leakage, "
                      "prompt format, or run `hf_fewshot --config <cfg> --debug`")

    pbar.close()


def reorder_output(output_file, data_file, id_key):
    data = load_jsonlines(data_file)
    data_ids = {d[id_key] for d in data}
    output = load_jsonlines(output_file)
    output_ids = {d[id_key] for d in output}

    assert data_ids == output_ids, "Data and output files have different IDs"
    output_dict = {d[id_key]: d for d in output}

    reordered_output = [output_dict[d[id_key]] for d in data]

    write_jsonlines(reordered_output, output_file)


def few_shot_classifier(config: dict):
    model_family = config["model_details"]["model_family"]
    model_name = config["model_details"]["model_name"]
    dynamic_batching = config["model_details"].get("dynamic_batching", False)

    prompt, exemplars = load_prompts_and_exemplars(config)
    outfile = prepare_output_file(config)
    query_texts, dataset, id_key = prepare_initial_data(config, outfile, exemplars, prompt)

    print(f"Generated {len(query_texts)} input prompts for {model_name}")

    model_params = config["model_details"]
    api_model = model_family == "gpt"

    if not api_model:
        try:
            display_gpu_status()
        except:
            print("could not call display_gpu_status")

    model_class = model_map[model_family]
    labels = config['prompt_details']['labels'] if 'labels' in config['prompt_details'] else None
    if labels:
        labels = parse_labels_config(labels)

    model = model_class(model_name=model_name, model_details=model_params, labels=labels)
    print("Model loaded")

    if not api_model:
        print("Model Loaded on GPU .. ")
        display_gpu_status()

    batch_size = model_params.get("batch_size", 1)
    id_values = [d[id_key] for d in dataset]

    run_inference(model, query_texts, batch_size, outfile, id_values, id_key, api_model, dynamic_batching)
    #reorder_output(outfile, config["dataset"]["path"], id_key)


def run_debug_checks(config: dict, num_samples: int = 5) -> None:
    """
    Pre-flight sanity checks for a config + model, run via `hf_fewshot --config ... --debug`.

    Validates the assumptions the scoring/preference pipeline relies on, so a new model
    fails loudly here instead of silently producing garbage scores:
      [1] each label is a single token for this tokenizer (scoring reads one position),
      [2] leading whitespace (space / newline / merged ws tokens) is skipped to reach the
          label, so " 5", "\\n5", " \\n5" all resolve to the label,
      [3] on real generations the model commits to a label at the first content token --
          this catches reasoning-mode <think> leakage and off-format answers (spelled-out
          words, prose), which show up as a high non-compliance rate.
    """
    model_family = config["model_details"]["model_family"]
    model_name = config["model_details"]["model_name"]
    api_model = model_family == "gpt"

    labels = config["prompt_details"].get("labels")
    if labels:
        labels = parse_labels_config(labels)
    if not labels:
        print("No labels configured in prompt_details; nothing to sanity-check.")
        return

    model = model_map[model_family](
        model_name=model_name, model_details=config["model_details"], labels=labels
    )

    print("\n" + "=" * 72)
    print("DEBUG SANITY CHECK")
    print("=" * 72)
    print(f"model  : {model_name} (family={model_family})")
    print(f"labels : {labels}")

    if api_model or not hasattr(model, "tokenizer"):
        print("\nTokenization checks apply to local HF models only "
              "(API models return string tokens directly). Skipping.")
        return

    tok = model.tokenizer
    problems: list[str] = []

    def is_whitespace(token_id: int) -> bool:
        return tok.decode([token_id]).strip() == ""

    # ---- [1] each label must be a single token ----
    print("\n[1] Label tokenization (each label must encode to a single token)")
    for label in labels:
        ids = tok.encode(label, add_special_tokens=False)
        ok = len(ids) == 1
        if not ok:
            problems.append(f"label {label!r} is multi-token {ids} -- scoring reads one position only")
        print(f"    {label!r:>6} -> {ids}  {'OK' if ok else 'PROBLEM (multi-token)'}")

    # ---- [2] leading-whitespace resilience ----
    print("\n[2] Leading-whitespace resilience (parser skips ws, lands on the label)")
    for label in labels:
        base = tok.encode(label, add_special_tokens=False)
        for prefix in (" ", "\n", " \n", "  "):
            variant = prefix + label
            ids = tok.encode(variant, add_special_tokens=False)
            kept = list(ids)
            while kept and is_whitespace(kept[0]):
                kept.pop(0)
            ok = kept[:1] == base[:1]
            if not ok:
                problems.append(f"variant {variant!r} -> {ids} does not resolve to label {label!r}")
            print(f"    {variant!r:>8} -> {str(ids):<14} first-content={tok.decode(kept[:1])!r:>6}  "
                  f"{'OK' if ok else 'PROBLEM'}")

    # ---- [3] live generation on a few real examples ----
    print(f"\n[3] Live generation on first {num_samples} dataset example(s)")
    try:
        prompt, exemplars = load_prompts_and_exemplars(config)
        dataset = load_records(config["dataset"]["path"], config["dataset"].get("format"))[:num_samples]
        output_var = config["prompt_details"]["output_var"]
        text_key = config["dataset"].get("text_key")
        query_texts = [prep_prompt(d, output_var, prompt=prompt, exemplars=exemplars, text_key=text_key) for d in dataset]

        out = model.generate_answer_batch_logprobs(query_texts)
        logprobs = get_logprobs(out["scores"])
        prefs = get_option_preferences(model, logprobs, labels)

        n_compliant = 0
        for ans, pref in zip(out["answers"], prefs):
            if pref is None:
                verdict = "NON-COMPLIANT (no label at first content token)"
            else:
                top = max(pref, key=pref.get)
                n_compliant += 1
                verdict = f"label={top}  p={pref[top]:.3f}"
            print(f"    output={ans!r:>14}  ->  {verdict}")

        n = len(prefs)
        rate = n_compliant / max(n, 1)
        print(f"\n    compliant: {n_compliant}/{n} ({rate:.0%})")
        if rate < 1.0:
            problems.append(f"{n - n_compliant}/{n} sample(s) non-compliant -- check reasoning-mode "
                            f"<think> leakage, prompt format, or max_new_tokens")
    except Exception as e:
        problems.append(f"live generation check failed to run: {e}")
        print(f"    could not run generation check: {e}")

    # ---- summary ----
    print("\n" + "=" * 72)
    if problems:
        print(f"FOUND {len(problems)} POTENTIAL ISSUE(S):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("All sanity checks passed.")
    print("=" * 72)


def get_args():
    parser = argparse.ArgumentParser(description="Run few-shot classification on a dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--debug", action="store_true",
                        help="Run tokenization/parsing sanity checks for the config's model "
                             "instead of full inference")
    parser.add_argument("--debug-samples", type=int, default=5,
                        help="Number of dataset examples to use for the --debug live-generation check")
    return parser

def main():
    parser = get_args()
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.debug:
        run_debug_checks(config, num_samples=args.debug_samples)
        return
    few_shot_classifier(config)


if __name__ == "__main__":
    main()
