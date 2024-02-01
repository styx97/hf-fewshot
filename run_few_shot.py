import yaml 
from tqdm.auto import tqdm
import json 
import argparse
from models import MistralFewShot
from prompting_utils import prep_prompt_new, load_yaml, load_jsonlines

# TODO: better way to import prompts? 
import prompts 

model_map = {
    "mistral": MistralFewShot,
}
# TODO: importing more models 

config = load_yaml("configs/mistral_fewshot_stance.yml")

def run_classifier(model_name: str,
                prompt: dict, 
                dataset: list[dict],
                batch_size: int):
    
    model = model_map[model_name]()
                            
    document_ids, document_texts = zip(*[(k, v['text']) for k, v in dataset.items()])
    
    query_texts = [prep_prompt_new(targets,
                                output_var,
                                prompt=prompt,
                                exemplar_path=exemplar_path) 
                                        for comment in document_texts]
    
    # debug
    #query_texts = query_texts[:100]

    print(f"Generated {len(query_texts)} input prompts for {model_name} on {topic}")

    generated_responses = {}
    pbar = tqdm(total=len(query_texts), desc="Generating responses")

    for i in range(0, len(query_texts), batch_size):
        pbar.update(batch_size)
        batch_ids = document_ids[i:i+batch_size]
        batch_query_texts = query_texts[i:i+batch_size]
        
        # this method should work across all models
        batch_responses, scores = model.generate_answer_batch_scores(batch_query_texts)
    
        for id, response, score in zip(batch_ids, batch_responses, scores):
            generated_responses[id] = {
                "response": response, 
                "scores": score
            }

    pbar.close()
    return generated_responses


def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", 
                        type=str, 
                        required=True)
    
    parser.add_argument("--dataset_path",
                        type=str,
                        required=False)
    

def main(args: argparse.Namespace, **kw): 
    
    config = load_yaml(args.config_file)
    model_name = config["model_details"]["model_name"]
    exemplars = load_jsonlines(config["exemplars"]["path"])
    prompt = getattr(prompts, config["prompts"]["prompt_name"])
    
    input_vars = config["prompts"]["input_vars"]
    output_var = config["prompts"]["output_var"]
    labels = config["prompts"]["labels"]

    
    """
    SANITY CHECK: input variables in the prompt + output variable should be 
    the same as the number of variables in the exemplar 
    """
    all_vars = set(input_vars + [output_var])
    exemplar_vars = set(exemplars[0].keys())
    assert all_vars == exemplar_vars, (f"Variables in the prompt: {all_vars} are not the"
                                    f"same as the variables in the exemplar: {exemplar_vars}")


    answers = run_classifier(model_name=model_name,
                        prompt=prompt,
                        dataset=dataset,
                        exemplar_path=exemplars, 
                        batch_size=config["model_details"]["batch_size"])
                                    

    with open(args.outfile, 'w') as f:
        json.dump(answers, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    main(args)




