from hf_fewshot.prompting_utils import load_jsonlines, write_jsonlines
import argparse
from pairwise_comparisons.template_strings import benoit_template_string

def label_pair_map(label):
    if label == 1: 
        return "A"
    elif label == 2:
        return "B"

def expand_pair(pair, template_str): 
    id_ = pair["pair_id"]
    text1 = pair["text1"]
    text2 = pair["text2"]
    label = pair["label"]
    reverse_label = 2 if label == 1 else 1
    
    prompt1 = template_str.format(textA=text1, textB=text2)
    prompt2 = template_str.format(textA=text2, textB=text1)

    return [{"id": f"{id_}.0", "prompt": prompt1, 'label': label_pair_map(label)}, 
            {"id": f"{id_}.1", "prompt": prompt2, 'label': label_pair_map(reverse_label)}]


def expand_dataset(dataset, transform_fn):
    new_dataset = []
    for pair in dataset:
        new_dataset.extend(transform_fn(pair))
    return new_dataset


def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", 
                        type=str, 
                        help="Path to the input dataset")
    
    parser.add_argument("--output_file", 
                        type=str, 
                        help="Path to the output dataset")
    
    parser.add_argument("--template_str", 
                        type=str, 
                        default=benoit_template_string, 
                        help="Template string to use for expansion")
    return parser

def main(parser):
    args = parser.parse_args()

    dataset = load_jsonlines(args.input_file)
    expanded_dataset = expand_dataset(dataset, lambda x: expand_pair(x, args.template_str))
    write_jsonlines(expanded_dataset, args.output_file)
    