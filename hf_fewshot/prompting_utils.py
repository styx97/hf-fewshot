import json, yaml

def load_jsonlines(filepath): 
    with open(filepath, 'r') as f: 
        data = [json.loads(line) for line in f if line.strip()]
    return data

# load a yaml file 
def load_yaml(filepath: str) -> dict: 
    with open(filepath, 'r') as f: 
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def load_json(filepath): 
    with open(filepath, 'r') as f: 
        data = json.load(f)
    return data

def write_json(data, filepath): 
    with open(filepath, 'w') as f: 
        json.dump(data, f, indent=4, sort_keys=True)

def write_jsonlines(data, filepath): 
    with open(filepath, 'w') as f: 
        for line in data: 
            json.dump(line, f)
            f.write('\n')


def prep_prompt(
        targets: dict,
        output_var: str, 
        prompt: dict,
        exemplars: list[dict]=None) -> list:
    
    
    """
    Takes a prompt, an exemplar path that is a jsonlines file, 
    and targets that contain the data to run predictions on. 

    Returns a list of messages that are expected by the 
    transformers 'apply_chat_template' method ()
    """

    beginner_prompt = prompt["zero_shot"]
    followup_prompts = prompt["followup"]

    if not exemplars:
        return [{
            "role": "user", 
            "content":  beginner_prompt.format(**targets)
        }]
    
    messages = [] 
    messages.append({
        "role": "user", 
        "content": beginner_prompt.format(**exemplars[0])
        })
    
    messages.append({
        "role": "assistant", 
        "content": exemplars[0][output_var]
    })

    for exemplar in exemplars[1:]:
        messages.append({
            "role": "user", 
            "content": followup_prompts.format(**exemplar)
        })
        messages.append({
            "role": "assistant", 
            "content": exemplar[output_var]
        })
    
    # finally, pass the target document
    messages.append({
        "role": "user", 
        "content": followup_prompts.format(**targets)
    })
    
    return messages

def prep_prompt_old(target_document: str, 
                         topic: str,
                         prompt: dict,
                         exemplar_path: str=None) -> list:
    
    beginner_prompt = prompt["zero_shot"]
    followup_prompts = prompt["followup"]

    if not exemplar_path:
        return [{
            "role": "user", 
            "content":  beginner_prompt.format(document=target_document,
                            topic=topic)
        }]
    
    exemplars = load_jsonlines(exemplar_path)
    exemplar_documents = [exemplar['text'] for exemplar in exemplars]
    exemplar_stances = [exemplar['stance'] for exemplar in exemplars]
    
    messages = [] 
    messages.append({
        "role": "user", 
        "content": beginner_prompt.format(document=exemplar_documents[0],
                                             topic=topic)
        })
    
    messages.append({
        "role": "assistant", 
        "content": exemplar_stances[0]
    })

    for document, stance in zip(exemplar_documents[1:], exemplar_stances[1:]): 
        messages.append({
            "role": "user", 
            "content": followup_prompts.format(document=document,
                                               topic=topic)
        })
        messages.append({
            "role": "assistant", 
            "content": stance
        })    
    
    # finally, pass the target document 
    messages.append({
        "role": "user", 
        "content": followup_prompts.format(document=target_document,
                                           topic=topic)
    })

    return messages