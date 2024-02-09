from tqdm.auto import tqdm
import json 
from pathlib import Path
from hf_fewshot.models import MistralFewShot
from hf_fewshot.prompting_utils import prep_prompt, load_yaml, load_jsonlines, load_json

model_map = {
    "mistral": MistralFewShot,
}
# TODO: add support for other models

def few_shot_classifier(config_file: str):

    config = load_yaml(config_file)
    model_family = config["model_details"]["model_family"]
    model_name = config["model_details"]["model_name"]
    
    exemplars = load_jsonlines(config["exemplars"]["path"])

    prompts = load_json(config["prompts"]["path"])
    prompt = prompts[config["prompts"]["prompt_name"]]
    batch_size = config["model_details"]["batch_size"]
    outfile = Path(config['output']['output_dir']) / f"{config['output']['task_name']}_on_{config['dataset']['dataset_name']}.jsonl"
    
    input_vars = config["prompts"]["input_vars"]
    output_var = config["prompts"]["output_var"]
    labels = config["prompts"]["labels"]
    
    dataset = load_jsonlines(config["dataset"]["path"])
    id_key = config["dataset"]["id_key"]
    all_id_values = [d[id_key] for d in dataset]

    assert id_key in dataset[0].keys(), f"ID key {id_key} not found in dataset"

    """
    SANITY CHECK: input variables in the prompt + output variable should be 
    the same as the number of variables in the exemplar 
    """
    all_vars = set(input_vars + [output_var])
    exemplar_vars = set(exemplars[0].keys())
    assert all_vars == exemplar_vars, (f"Variables in the prompt: {all_vars} are not the"
                                    f"same as the variables in the exemplar: {exemplar_vars}")

    all_dataset_vars = set(dataset[0].keys())
    # Check that all variables in the prompt are in the dataset
    assert set(input_vars).issubset(all_dataset_vars), (f"Variables in the prompt: {input_vars} are not "
                                                    f"found in the dataset: {all_dataset_vars}")

    model = model_map[model_family](model_name=model_name,
                                labels=labels, 
                                model_details=config["model_details"])
    
    query_texts = [prep_prompt(targets,
                                output_var,
                                prompt=prompt,
                                exemplars=exemplars) 
                                        for targets in dataset]
    

    print(f"Generated {len(query_texts)} input prompts for {model_name}")
    pbar = tqdm(total=len(query_texts), desc="Generating responses")
    
    with open(outfile, 'w') as f:
        print("Writing responses to ", config["outfile"])
        for i in range(0, len(query_texts), batch_size):
            pbar.update(batch_size)
            batch_query_texts = query_texts[i:i+batch_size]
            ids = all_id_values[i:i+batch_size]
            batch_responses, scores = model.generate_answer_batch_scores(batch_query_texts)
            assert len(batch_responses) == len(scores), "Batch size mismatch"
            
            for id_, response, score in zip(ids, batch_responses, scores):
                f.write(json.dumps({id_key: id_, "response": response, "scores": score}))
                f.write('\n')
    
    pbar.close()


few_shot_classifier("/fs/clip-political/rupak/opinion_multivalence/configs/mistral_fewshot_stance.yml")