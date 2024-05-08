from tqdm.auto import tqdm
import json 
from pathlib import Path
from hf_fewshot.models import MistralFewShot, HFFewShot, LlamaFewShot, GPTFewShot
from hf_fewshot.prompting_utils import prep_prompt, load_yaml, load_jsonlines, load_json
import numpy as np
import os 
from dotenv import load_dotenv
import argparse

# update HF and OPENAI_TOKENS 
load_dotenv("../../.env")


model_map = {
    "mistral": MistralFewShot,
    "hf-general": HFFewShot,
    "llama": LlamaFewShot, 
    "gpt": GPTFewShot
}
# TODO: add support for other models

def few_shot_classifier(config_file: str):

    config = load_yaml(config_file)
    model_family = config["model_details"]["model_family"]
    model_name = config["model_details"]["model_name"]
    
    exemplars_path = config["exemplars"]["path"]
    
    exemplars = load_jsonlines(exemplars_path) if exemplars_path !='None' else None
    
    shuffle_exemplars = config["exemplars"]["shuffle"]
    num_exemplars = config["exemplars"]["num_exemplars"]

    prompts = load_json(config["prompts"]["path"])
    prompt = prompts[config["prompts"]["prompt_name"]]
    batch_size = config["model_details"]["batch_size"]
    
    if config["output"]["output_file"]: 
        outfile = Path(config["output"]["output_dir"]) / config["output"]["output_file"]
    else: 
        outfile = Path(config['output']['output_dir']) / f"{config['output']['task_name']}_on_{config['dataset']['dataset_name']}_{model_name}.jsonl"

    # if the output file doesn't exist, create the intermediate directiories and the file
    if not outfile.is_file():
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.touch()
        
    input_vars = config["prompts"]["input_vars"]
    output_var = config["prompts"]["output_var"]
    #labels = config["prompts"]["labels"]
    
    dataset = load_jsonlines(config["dataset"]["path"])
    id_key = config["dataset"]["id_key"]
    all_id_values = [d[id_key] for d in dataset]

    assert id_key in dataset[0].keys(), f"ID key {id_key} not found in dataset"

    """
    SANITY CHECK: input variables in the prompt + output variable should be 
    the same as the number of variables in the exemplar 
    """
    all_vars = set(input_vars + [output_var])
    if exemplars:
        exemplar_vars = set(exemplars[0].keys())
        assert all_vars == exemplar_vars, (f"Variables in the prompt: {all_vars} are not the"
                                        f"same as the variables in the exemplar: {exemplar_vars}")

    all_dataset_vars = set(dataset[0].keys())
    # Check that all variables in the prompt are in the dataset
    assert set(input_vars).issubset(all_dataset_vars), (f"Variables in the prompt: {input_vars} are not "
                                                    f"found in the dataset: {all_dataset_vars}")

    # shuffle the exemplars here, run all prompts on the same order 
    if shuffle_exemplars:
        np.random.seed(42)
        np.random.shuffle(exemplars)
    
    # take the first num_exemplars
    exemplars = exemplars[:num_exemplars]
    
    query_texts = [prep_prompt(targets,
                                output_var,
                                prompt=prompt,
                                exemplars=exemplars, 
                                ) 
                                        for targets in dataset]
    

    if outfile.is_file():
        existing_data = load_jsonlines(outfile)
        if len(existing_data) > 0:
            print("Found existing data in ", outfile)
            print("setting start index to ", len(existing_data))
            start_index = len(existing_data)
        else: 
            start_index = 0
    else:
        start_index = 0

    print(f"Start index is: {start_index}")    

    # model loading 
    model_details = config["model_details"]
    model = model_map[model_family](model_name=model_name,
                                model_details=model_details)
    print("Model loaded")
    
    print(f"Generated {len(query_texts)} input prompts for {model_name}")
    
    pbar = tqdm(total=len(query_texts) - start_index, desc="Generating responses")

    with open(outfile, "a+") as f:
        print("Loading outfile")
        # If not starting from zero, first add a newline 
        
        print("Writing responses to ", outfile)
        for i in range(start_index, len(query_texts), batch_size):
            pbar.update(batch_size)
            batch_query_texts = query_texts[i:i+batch_size]
            ids = all_id_values[i:i+batch_size]

            if model_details["scores"]: 
                batch_responses, scores = model.generate_answer_batch_scores(batch_query_texts)
                assert len(batch_responses) == len(scores), "Batch size mismatch"
                for id_, response, score in zip(ids, batch_responses, scores):
                    f.write(json.dumps({id_key: id_, "response": response, "scores": score}))
                    f.write('\n')

            else:
                batch_responses = model.generate_answer_batch(batch_query_texts)
                for id_, response in zip(ids, batch_responses):
                    f.write(json.dumps({id_key: id_, "response": response}))
                    f.write('\n')
    
    pbar.close()



def get_args():
    parser = argparse.ArgumentParser(description="Run few-shot classification on a dataset")
    parser.add_argument("--config", 
                        type=str, 
                        help="Path to the config file")

    return parser

def main(): 
    parser = get_args()
    
    args = parser.parse_args()
    few_shot_classifier(args.config)