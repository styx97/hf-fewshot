from tqdm.auto import tqdm
import json
from pathlib import Path
import numpy as np
import re

import argparse

import torch
from hf_fewshot.models import (
    MistralFewShot,
    HFFewShot,
    LlamaFewShot,
    GPTFewShot,
    GemmaFewShot,
    Gwen2FewShot,
    Gemma3FewShot,
    display_gpu_status,
    get_unused_gpu_memory, 
    get_logsoftmax
)

from hf_fewshot.prompting_utils import (
    prep_prompt,
    load_yaml,
    load_jsonlines,
    load_json,
    write_jsonlines, 
    read_md
)

model_map = {
    "mistral": MistralFewShot,
    "hf-general": HFFewShot,
    "llama": LlamaFewShot,
    "gpt": GPTFewShot,
    "gemma": GemmaFewShot,
    "gemma3": Gemma3FewShot,
    "qwen2": Gwen2FewShot
}

def get_option_preferences(model: LlamaFewShot, 
                           logprobs: np.array, 
                           options: list[str]) -> np.array: 
    """
    Given the logprobs of the options, find the model preferences for each option
    """

    # NOTE: keep all tokens for multi-token options
    option_token_dict = {
        option: model.tokenizer.encode(option, add_special_tokens=False) 
        for option in options
    }
    
    preferences = []
    num_examples = logprobs.shape[0]

    for index in range(num_examples):
        option_logprobs  = {
            option: logprobs[index, 0, option_token_dict[option]] 
            for option in options
        }
        # convert logprobs to probabilities 

        # NOTE: consider joint probability for multi-token options
        option_probs = {
            option: float(np.exp(logprob.sum())) 
            for option, logprob in option_logprobs.items()
        }

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

    Accepts two ways of prompt input:
    1. A json file that contains the prompt with the keys "zero_shot" and "followup"
    2. A markdown directory that contains the prompt in the files "zero_shot.md" and "followup.md"
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
        
    if config['exemplars']['use_exemplars']:
        exemplars_path = config["exemplars"]["path"]
        exemplars = load_jsonlines(exemplars_path) if exemplars_path != 'None' else None
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

    dataset = load_jsonlines(config["dataset"]["path"])
    id_key = config["dataset"]["id_key"]
    input_vars = config["prompt_details"]["input_vars"]
    output_var = config["prompt_details"]["output_var"]

    assert id_key in dataset[0].keys(), f"ID key {id_key} not found in dataset"
    assert set(input_vars).issubset(set(dataset[0].keys())), \
        (f"Variables in the prompt: {input_vars} are not found in the dataset: {set(dataset[0].keys())}")

    if exemplars:
        all_vars = set(input_vars + [output_var])
        exemplar_vars = set(exemplars[0].keys())
        assert set(all_vars).issubset(exemplar_vars), \
            (f"Variables in the prompt: {all_vars} are not the same as the variables in the exemplar: {exemplar_vars}")

    existing_data = load_jsonlines(outfile)
    if existing_data:
        existing_ids = {d[id_key] for d in existing_data}
        dataset = [d for d in dataset if d[id_key] not in existing_ids]
        print(f"Found {len(existing_data)} data items in the output file.")
        print(f"Running inference on {len(dataset)} items.")

    query_texts = [prep_prompt(d, output_var, prompt=prompt, exemplars=exemplars) for d in dataset]
    
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
    
    with open(outfile, "a+") as f:
        i = 0
        while i < len(query_texts):
            try:
                batch_query_texts = query_texts[i:i + batch_size]
                ids = id_values[i:i + batch_size]
                batched_output = model.generate_answer_batch_logprobs(batch_query_texts)
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
                    f.write(json.dumps(output) + "\n")
                    
                i += batch_size
                pbar.update(batch_size)

                if api_model:
                    continue

                unused_gpu_mem = get_unused_gpu_memory()
                print("Unused GPU memory: ", unused_gpu_mem)

                # if dynamic batching is turned on, increase batch size by 2 
                if unused_gpu_mem > 40 and dynamic_batching:
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


def few_shot_classifier(config_file: str):
    config = load_yaml(config_file)
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


def get_args():
    parser = argparse.ArgumentParser(description="Run few-shot classification on a dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    return parser

def main():
    parser = get_args()
    args = parser.parse_args()
    few_shot_classifier(args.config)



if __name__ == "__main__": 
    main()