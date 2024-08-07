"""
Implements basic debiasing: 

1. Finds the logprobs of the options for each option in a batch 
2. Using the logprobs, finds model preferences for each option in that batch 
3. Using model preferences, adjusts the thresholds for deciding a win 
"""

from hf_fewshot.models import get_logsoftmax, LlamaFewShot 
from hf_fewshot.prompting_utils import load_jsonlines, write_jsonlines
import numpy as np 
from tqdm import tqdm 

def generate_answer_batch_logprobs(model_obj: LlamaFewShot, 
                        query_texts: list) -> list[str]: 
    
    """
    Code to batch process multiple questions. 
    Can be generalized to other types of query processing.
    """
    messages = [
        model_obj.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        ) for messages in query_texts
    ]

    model_inputs = model_obj.tokenizer(
        messages,
        return_tensors="pt", 
        padding=True
    ).to("cuda")

    terminators = [
        model_obj.tokenizer.eos_token_id,
        model_obj.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model_obj.model.generate(
        **model_inputs, 
        max_new_tokens = 4,
        do_sample=False,
        #temperature=0.01,
        eos_token_id=terminators,
        return_dict_in_generate=True, 
        pad_token_id=model_obj.tokenizer.eos_token_id,
        output_scores=True
    )


    answer_texts = model_obj.tokenizer.batch_decode(outputs.sequences[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    scores = outputs.scores
    
    # return the scores and answers
    return {"answers": answer_texts,
             "scores": scores}



def get_logprobs(scores): 
    """
    The shape of scores as returned by the model is (max_new_tokens, batch_size, vocab_size)

    This function takes raw logit scores and returns logprobs in the shape (batch_size, max_new_tokens, vocab_size)
    """

    # find out the batch size 
    batch_size = scores[0].shape[0]
    print(batch_size)
    logprobs = [] 
    for index in range(batch_size):
        curr_logprobs = []
        for token_output in scores: 
            token_output_logprobs = get_logsoftmax(token_output[index]) 
            curr_logprobs.append(token_output_logprobs[0].detach().cpu().numpy())
        
        logprobs.append(curr_logprobs)
    
    return np.array(logprobs)


def get_option_preferences(model: LlamaFewShot, 
                           logprobs: np.array, 
                           options: list[str]) -> np.array: 
    """
    Given the logprobs of the options, find the model preferences for each option
    """

    option_token_dict = {option: model.tokenizer.encode(option)[1] for option in options}
    
    preferences = []
    num_examples = logprobs.shape[0]

    for index in range(num_examples):
        option_logprobs  = {option: logprobs[index, 0, option_token_dict[option]] 
                            for option in options}
        option_logprob_ratios = {option: option_logprobs[option]/np.sum(list(option_logprobs.values())) 
                                 for option in options}

        preferences.append(option_logprob_ratios)

    return preferences



if __name__ == "__main__": 

    model_details = {
    'quantization': '4bit',
    'max_new_tokens': 4,
    'temperature': 0.01,
    'do_sample': False,
    'return_dict_in_generate': True,
    'model_family': 'llama',
    'model_name': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'scores': False,
    'batch_size': 8,
    }

    model_name = model_details["model_name"]
    model = LlamaFewShot(model_name, model_details)

    # load the data
    benoit_pairs = load_jsonlines("data/0.train.expanded.jsonl")
    print(f"Loaded {len(benoit_pairs)} pairs")

    # define data parameters:
    options = ["A", "B"]

    # benoit paired messages : 
    messages_paired = [
         [
            {"role": "system", "content": "You are a helpful digital assistant. You will be asked to compare two items A and B on a given criteria. Reply only with A or B."},
            {"role": "user", "content": pair['prompt']},
        ] for pair in benoit_pairs
    ]

    # generate answers and scores 
    batch_size = model_details["batch_size"]

    batched_outputs = [] 
    for index in tqdm(range(0, len(messages_paired), batch_size)):
        batched_output = generate_answer_batch_logprobs(model, messages_paired[index:index+batch_size])
        batched_outputs.append(batched_output)
        logprobs = get_logprobs(batched_output["scores"])
        preferences = get_option_preferences(model, logprobs, options)

        batched_outputs.append({
            "output": batched_output["answers"],
            "preferences": preferences
        })

    write_jsonlines(batched_outputs, "data/0.train.expanded.output.jsonl")

