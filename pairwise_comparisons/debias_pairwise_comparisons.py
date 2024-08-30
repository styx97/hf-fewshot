"""
Implements basic debiasing: 

1. Finds the logprobs of the options for each option in a batch 
2. Using the logprobs, finds model preferences for each option in that batch 
3. Using model preferences, adjusts the thresholds for deciding a win 
"""

from hf_fewshot.models import get_logsoftmax, LlamaFewShot 
from hf_fewshot.prompting_utils import load_jsonlines, write_jsonlines
import numpy as np 
import argparse
from tqdm import tqdm 
from torch import cuda

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
        
        # convert logprobs to probabilities 
        option_probs = {option: float(np.exp(logprob)) for option, logprob in option_logprobs.items()}

        preferences.append(option_probs)

    return preferences


def add_args(): 
    parser = argparse.ArgumentParser(description="Debiasing pairwise comparisons")
    parser.add_argument("--input", type=str, help="Input file path", required=True)
    parser.add_argument("--output", type=str, help="Output file path", required=True)
    parser.add_argument("--model_name", type=str, help="Model name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    
    return parser 


def prep_prompts(pair_data: dict) -> list: 
    sep = "\n\n"
    messages = [
        {
            "role": "system",
            "content": "You are helpful digital assistant. You will be asked to compare two text items - Text 1 and Text 2 based on a criteria. Reply only with 1 or 2."
        },
    ]

    # first, add the instruction with the first message 
    instruction = pair_data["prompt"]['instructions']
    message_content = pair_data["prompt"]["message_contents"]
    messages.append({
        "role": "user",
        "content": instruction + sep + sep.join(message_content[0][:2])
    })

    messages.append({
        "role": "assistant",
        "content": message_content[0][2]
    })

    # for the rest of the messages except the last one, add the followup prompts
    for index in range(1, len(message_content)-1):
        messages.append({
            "role": "user",
            "content": sep.join(message_content[index][:2])
        })

        messages.append({
            "role": "assistant",
            "content": message_content[index][2]
        })

    # add the last message
    messages.append({
        "role": "user",
        "content": message_content[-1] + "\n" + pair_data["prompt"]["ending"]
    })

    return messages


def compute_scores(file_path: str, output_path: str, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"): 

    model_details = {
        'quantization': '4bit',
        'max_new_tokens': 4,
        'temperature': 0.01,
        'do_sample': False,
        'return_dict_in_generate': True,
        'model_family': 'llama',
        'model_name': model_name,
        'scores': False,
        'batch_size': 8,
    }

    model_name = model_details["model_name"]
    model = LlamaFewShot(model_name, model_details)

    # load the data
    pairs = load_jsonlines(file_path)
    print(f"Loaded {len(pairs)} pairs")

    # define data parameters:
    options = ["1", "2"]

    # benoit paired messages : 
    messages_paired = [prep_prompts(pair) for pair in pairs]
    
    # debug script - focus on the first k batches 
    # messages_paired = messages_paired[:64]

    # generate answers and scores 
    batch_size = model_details["batch_size"]

    batched_outputs = [] 
    
    for index in tqdm(range(0, len(messages_paired), batch_size)):
        batched_output = generate_answer_batch_logprobs(model, messages_paired[index:index+batch_size])
        
        logprobs = get_logprobs(batched_output["scores"])
        preferences = get_option_preferences(model, logprobs, options)

        # save the outputs one by one 
        for index, preference, answer in zip(range(index, index+batch_size), preferences, batched_output["answers"]):
            batched_outputs.append({
                "pair_id": pairs[index]["pair_id"],
                "output": answer,
                "preferences": preference, 
                "label": pairs[index]["label"],
            })

        #clear the cache in gpu
        #cuda.empty_cache()

        
    write_jsonlines(batched_outputs, output_path)

if __name__ == "__main__": 
    parser = add_args()
    args = parser.parse_args()

    """
    python debias_pairwise_comparisons.py --input ../../pairwise-comparison/experiments-2024/benoit2019_2024-08-29/intermediate/draw-0.jsonl --output output/draw-0-results.jsonl; python debias_pairwise_comparisons.py --input ../../pairwise-comparison/experiments-2024/benoit2019_2024-08-29/intermediate/draw-1.jsonl --output output/draw-1-results.jsonl; python debias_pairwise_comparisons.py --input ../../pairwise-comparison/experiments-2024/benoit2019_2024-08-29/intermediate/draw-2.jsonl --output output/draw-2-results.jsonl
    
    python debias_pairwise_comparisons.py --input ../../pairwise-comparison/experiments-2024/benoit2019_2024-08-29/intermediate/draw-0.jsonl --output output/draw-0-results_70B.jsonl --model_name "meta-llama/Meta-Llama-3-70B-Instruct"; python debias_pairwise_comparisons.py --input ../../pairwise-comparison/experiments-2024/benoit2019_2024-08-29/intermediate/draw-1.jsonl --output output/draw-1-results_70B.jsonl --model_name "meta-llama/Meta-Llama-3-70B-Instruct"; python debias_pairwise_comparisons.py --input ../../pairwise-comparison/experiments-2024/benoit2019_2024-08-29/intermediate/draw-2.jsonl --output output/draw-2-results_70B.jsonl --model_name "meta-llama/Meta-Llama-3-70B-Instruct"
    
    
    """

    compute_scores(args.input, args.output, args.model_name)