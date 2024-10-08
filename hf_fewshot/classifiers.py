from tqdm.auto import tqdm
import json
from pathlib import Path
import numpy as np
import argparse
import torch

from hf_fewshot.models import (
    MistralFewShot,
    HFFewShot,
    LlamaFewShot,
    GPTFewShot,
    GemmaFewShot,
    display_gpu_status,
    get_unused_gpu_memory
)
from hf_fewshot.prompting_utils import (
    prep_prompt,
    load_yaml,
    load_jsonlines,
    load_json,
    write_jsonlines
)

model_map = {
    "mistral": MistralFewShot,
    "hf-general": HFFewShot,
    "llama": LlamaFewShot,
    "gpt": GPTFewShot,
    "gemma": GemmaFewShot
}


def load_prompts_and_exemplars(config: dict) -> tuple[str, list[dict]]:
    """
    Load the prompt and exemplars from the config file 
    If no exemplars are provided, return None
    """

    prompts = load_json(config["prompts"]["path"])
    prompt = prompts[config["prompts"]["prompt_name"]]
    if "exemplars" in config:
        exemplars_path = config["exemplars"]["path"]
        exemplars = load_jsonlines(exemplars_path) if exemplars_path != 'None' else None
        if config["exemplars"]["shuffle"]:
            np.random.seed(42)
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
    input_vars = config["prompts"]["input_vars"]
    output_var = config["prompts"]["output_var"]

    assert id_key in dataset[0].keys(), f"ID key {id_key} not found in dataset"
    assert set(input_vars).issubset(set(dataset[0].keys())), \
        (f"Variables in the prompt: {input_vars} are not found in the dataset: {set(dataset[0].keys())}")

    if exemplars:
        all_vars = set(input_vars + [output_var])
        exemplar_vars = set(exemplars[0].keys())
        assert all_vars == exemplar_vars, \
            (f"Variables in the prompt: {all_vars} are not the same as the variables in the exemplar: {exemplar_vars}")

    existing_data = load_jsonlines(outfile)
    if existing_data:
        existing_ids = {d[id_key] for d in existing_data}
        dataset = [d for d in dataset if d[id_key] not in existing_ids]
        print(f"Found {len(existing_data)} data items in the output file.")
        print(f"Running inference on {len(dataset)} items.")

    query_texts = [prep_prompt(d, output_var, prompt=prompt, exemplars=exemplars) for d in dataset]
    
    return query_texts, dataset, id_key


def run_inference(model, query_texts, batch_size, outfile, id_values, id_key, api_model):
    print("Starting inference loop")
    pbar = tqdm(total=len(query_texts), desc='Running Inference')
    
    with open(outfile, "a+") as f:
        i = 0
        while i < len(query_texts):
            try:
                batch_query_texts = query_texts[i:i + batch_size]
                ids = id_values[i:i + batch_size]
                batch_responses = model.generate_answer_batch(batch_query_texts)
                for id_, response in zip(ids, batch_responses):
                    f.write(json.dumps({id_key: id_, "response": response}) + '\n')

                i += batch_size
                pbar.update(batch_size)

                if api_model:
                    continue

                unused_gpu_mem = get_unused_gpu_memory()
                print("Unused GPU memory: ", unused_gpu_mem)

                if unused_gpu_mem > 40:
                    batch_size += 2
                    print(f"Increasing batch size to {batch_size}")
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as e:
                print("Out of memory error. Reducing batch size")
                print(e)
                display_gpu_status()

                batch_size = max(1, batch_size - 2)
                print(f"Reducing batch size to {batch_size}")
                torch.cuda.empty_cache()

            if not api_model:
                torch.cuda.empty_cache()

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

    prompt, exemplars = load_prompts_and_exemplars(config)
    outfile = prepare_output_file(config)
    query_texts, dataset, id_key = prepare_initial_data(config, outfile, exemplars, prompt)

    print(f"Generated {len(query_texts)} input prompts for {model_name}")

    model_params = config["model_details"]
    api_model = model_family == "gpt"

    if not api_model:
        display_gpu_status()

    model_class = model_map[model_family]
    model = model_class(model_name=model_name, model_details=model_params)
    print("Model loaded")

    if not api_model:
        print("Model Loaded on GPU .. ")
        display_gpu_status()

    batch_size = model_params.get("batch_size", 1)
    id_values = [d[id_key] for d in dataset]

    run_inference(model, query_texts, batch_size, outfile, id_values, id_key, api_model)
    reorder_output(outfile, config["dataset"]["path"], id_key)


def get_args():
    parser = argparse.ArgumentParser(description="Run few-shot classification on a dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")

    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    few_shot_classifier(args.config)
