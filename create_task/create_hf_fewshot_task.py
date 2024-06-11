#################################################################
# python create_hf_fewshot_task.py output_dir output_prefix
#
# Input arguments:
#
#    --output_dir    output_dir
#
#      Directory, e.g. /Users/somebody/work/my_task_files
#
#      If this argument is a relative rather than absolute path for output_dir,
#      the program converts it into a full absolute path assuming that
#      the first part of the path leads to the current working directory,
#      with an informative report to stderr that it is doing so.
#
#    --output_prefix output_prefix   (defaults to 'test')
#
#      If the default is used, there is an informative report to stderr saying so.
#   
#    --input_label   input_label     (defaults to 'question')
#   
#    --output_label  output_label    (defaults to 'answer')
#
#    --prompt_name   prompt_name     (defaults to 'my_prompt')
#
#    --id_key        element name for data item IDs (defaults to 'ID')
#
#    --model_details YAML file, e.g. model_details_gpt.yaml 
#   
#    --instruction   text file, e.g. instruction.txt
#
#    --begin         string delimiting start of answer (defaults to '[begin]')
#
#    --end           string delimiting end of answer (defaults to '[end]')
#
#    --followup      string, e.g. "followup string"
#   
#      String for followup to instruction. .
#      This will be appended to the instruction, and also used as the 'followup' element for the prompt
#      Example with --input_label='question', --output_label='answer':
#
#        --followup_string 'Question: {question}\n\nAnswer:[begin]answer[end]'
#   
#    --exemplars    file with exemplars, e.g. exemplars.csv
#   
#      CSV file with column headers <id_key>, <input_label>, <output_label>
#      Rows contain exemplars. Example:
#
#        ID, question, answer
#        1,  What is the tallest mountain in the world?, [begin]Mount Everest[end]
#        2,  Which planet is known as the Red Planet?, [begin]Mars[end]
#
#      Note that values for <id_key> should be treated as strings even if they appear numeric
#
#    --dataset_name   name of dataset, e.g. qa_dataset_2024-05-24
#
#    --dataset      file with test items dataset, e.g. dataset.csv
#
#      CSV file with three columns containing test items: ID, input, output
#      The entire output column is optional, and even if it's present the individual output values are also optional
#
#    --task_output_dir   directory  (defaults to ./<output_prefix> if unspecified)
#
#      Directory, e.g. /Users/somebody/work/task_out
#      This will appear in the config file that gets created
#
#      If this argument is a relative rather than absolute path for task_output_dir,
#      the program converts it into a full absolute path assuming that
#      the first part of the path leads to the current working directory,
#      with an informative report to stderr that it is doing so.
#
# This program creates the following files in output files in output_dir,
# creating that directory if necessary. Make sure that any strings inside json or jsonl
# elements escape special characters appropriately.
# 
#   File <output_prefix>.prompt.json
#   
#     This output json file should contain the following:
# 
#     {
#       "<prompt_name>": {
#           "zero_shot": "<contents of instruction.txt>\n#####\n<contents of followup string> ",
#           "followup":  "<contents of followup string>"
#       }
#     }
# 
#     For example:
# 
#     {
#       "question_answering_prompt": {
#           "zero_shot": "You are a prompt following intelligent AI assistant that answers questions with an answer in the format below.\n#####\nQuestion: {question}\n\nAnswer:[begin]answer goes here[end] ",
#           "followup":  "Question: {question}\n\nAnswer:[begin]answer goes here[end]"
#       }
#     }
# 
#   File <output_prefix>.exemplars.jsonl
# 
#     This output jsonlines file should contain one jsonline for each exemplar in exemplars.csv.
#     The two elements should correspond to input_label and output_label.
#     For example:
# 
#         {"question": "What is the tallest mountain in the world?", "answer": "[begin]Mount Everest[end]"}
#         {"question": "Which planet is known as the Red Planet?", "answer": "[begin]Mars[end]"}
# 
#   File <output_prefix>.data.jsonl
# 
#     This output jsonlines file should contain one jsonline for each of the rows in dataset.csv.
#     For example, consider dataset.csv containing
# 
#         0, "Who is the current prime minister of England?", "Rishi Sunak"
#         1, "What is the tallest mountain in the world?"
#         2, "Which country has the largest population?", "China"
# 
#     In each jsonline first element with the ID should be <id_key>
#     The second and third elements should correspond to input_label and output_label
#     The content for output_label should be bracketed by [begin]...[end]
#     So for this dataset.csv the output file would contain:
# 
#         {"ID": "0", "question": "Who is the current prime minister of England?", "answer": "[begin]Rishi Sunak[end]"}
#         {"ID": "1", "question": "What is the tallest mountain in the world?", "answer": "[begin][end]"}
#         {"ID": "2", "question": "Which country has the largest population?", "answer": "[begin]China[end]"}
#       
#     Note that values of the <id_key> element should be treated as strings even if they appear numeric.
#     
# 
#   File <output_prefix>.config.yaml
# 
#     This is a YAML config file. It will contain the following, where material in <...> refers
#     to variables discussed above or material constructed using those variables. Material that's not
#     inside <...> is included verbatim.
# 
#         # Main header
#         <insert contents of <model_details> file>
# 
#         exemplars:
#             path: <full path for <output_prefix>.exemplars.jsonl>
#             num_exemplars: <count of exemplars in that file>
#             shuffle: False
# 
#         prompts: 
#             path: <full path for <output_prefix>.prompt.json>
#             prompt_name: <prompt_name>
#             output_var: <output_label>
#             labels:  # keep this empty for open labels
#             input_vars: [<input_label>]
# 
#         # The dataset should be a jsonlines file with keys that are of the following type 
#         dataset:
#             dataset_name: <dataset_name>
#             id_key: <id_key>
#             path: <full path for <dataset>.jsonl>
# 
#         output:
#             output_dir: <task_output_dir>
#             output_file: <task_name>.output.jsonl
#             task_name: <task_name>
#
# Written using ChatGPT 4o, 2024-05-31, plus minor editing
#
################################################################# 
import os
import sys
import json
import csv
import yaml
import pprint

def main():
    # Parse command line arguments
    args = parse_args()
    args_pretty = pprint.pformat(vars(args), indent=4, sort_dicts=False)
    sys.stderr.write(f"Running with arguments:\n{args_pretty}\n")

    
    # Handle directory paths
    output_dir = os.path.abspath(args.output_dir)
    task_output_dir = os.path.abspath(args.task_output_dir) if args.task_output_dir else os.path.join(output_dir, args.output_prefix)

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Process input files and create the necessary output files
    create_prompt_file(args, output_dir)
    create_exemplars_file(args, output_dir)
    create_data_file(args, output_dir)
    create_config_file(args, output_dir, task_output_dir)

    # Provide command line to run
    print(f"To run: hf_fewshot --config {output_dir}/{args.output_prefix}.config.yaml")

    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Create task files for hf_fewshot")
    
    parser.add_argument("--output_dir", required=True, help="Directory for hf_fewshot task files created by this program, e.g. the config file")
    parser.add_argument("--output_prefix", default="test", help="Prefix for hf_fewshot task filenames")
    parser.add_argument("--id_key", default="ID", help="Column name for IDs (defaults to 'ID')")
    parser.add_argument("--input_label", default="question", help="Column name for input items (defaults to 'question')")
    parser.add_argument("--output_label", default="answer", help="Column name for output items (defaults to 'answer')")
    parser.add_argument("--model_details", required=True, help="YAML file with model details")
    parser.add_argument("--prompt_name", default="my_prompt", help="Name of the prompt (defaults to 'my_prompt')")
    parser.add_argument("--instruction", required=True, help="Text file containing instruction part of the prompt")
    parser.add_argument("--begin", default="[begin]", help="String delimiting start of output (defaults to '[begin]')")
    parser.add_argument("--end",  default="[end]", help="String delimiting end of output (defaults to '[end']")
    parser.add_argument("--followup", required=True, help="String for followup to instruction (should contain the begin/end delimiters in outputs)")
    parser.add_argument("--exemplars", required=True, help="CSV file with exemplars (id_key, input_label, output_label)")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument("--dataset", required=True, help="CSV file with test items dataset (id_key, input_label, output_label); outputs optional")
    parser.add_argument("--task_output_dir", help="Directory where LLM task output should go when hf_fewshot is called with this config")
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
        sys.stderr.write(f"Converted output_dir to absolute path: {args.output_dir}\n")
    
    if not os.path.isabs(args.task_output_dir) and args.task_output_dir:
        args.task_output_dir = os.path.abspath(args.task_output_dir)
        sys.stderr.write(f"Converted task_output_dir to absolute path: {args.task_output_dir}\n")
    
    if args.output_prefix == "test":
        sys.stderr.write("Using default output_prefix 'test'\n")
    
    return args

def create_prompt_file(args, output_dir):
    sys.stderr.write("Creating prompt file\n")
    if args.begin and args.begin not in args.followup:
        sys.stderr.write(f"Warning: '{args.begin}' not found in followup string '{args.followup}'\n")
    if args.end and args.end not in args.followup:
        sys.stderr.write(f"Warning: '{args.end}' not found in followup string '{args.followup}'\n")
    instruction_text = read_file(args.instruction)
    prompt_content = {
        args.prompt_name: {
            "zero_shot": f"{instruction_text}\n#####\n{args.followup}",
            "followup": args.followup
        }
    }
    output_path = os.path.join(output_dir, f"{args.output_prefix}.prompt.json")
    with open(output_path, 'w') as f:
        json.dump(prompt_content, f, indent=4)
    
def create_exemplars_file(args, output_dir):
    sys.stderr.write("Creating exemplars file\n")
    output_path = os.path.join(output_dir, f"{args.output_prefix}.exemplars.jsonl")
    with open(output_path, 'w') as f_out:
        with open(args.exemplars, 'r') as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                if args.begin and args.begin not in row[args.output_label]:
                    sys.stderr.write(f"Warning: '{args.begin}' not found in {args.output_label} '{row[args.output_label]}'\n")
                if args.end and args.end not in row[args.output_label]:
                    sys.stderr.write(f"Warning: '{args.end}' not found in {args.output_label} '{row[args.output_label]}'\n")
                exemplar = {
                    args.input_label: row[args.input_label],
                    args.output_label: row[args.output_label]
                }
                f_out.write(json.dumps(exemplar) + "\n")
    
def create_data_file(args, output_dir):
    sys.stderr.write("Creating test data file\n")
    output_path = os.path.join(output_dir, f"{args.output_prefix}.data.jsonl")
    with open(output_path, 'w') as f_out:
        with open(args.dataset, 'r') as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                if args.begin and args.begin not in row[args.output_label]:
                    sys.stderr.write(f"Warning: '{args.begin}' not found in {args.output_label} '{row[args.output_label]}'\n")
                if args.end and args.end not in row[args.output_label]:
                    sys.stderr.write(f"Warning: '{args.end}' not found in {args.output_label} '{row[args.output_label]}'\n")
                data_item = {
                    args.id_key: row[args.id_key],
                    args.input_label: row[args.input_label],
                    args.output_label: f"{args.begin}{row[args.output_label]}{args.end}" if row[args.output_label] else f"{args.begin}{args.end}"
                }
                f_out.write(json.dumps(data_item) + "\n")

def create_config_file(args, output_dir, task_output_dir):
    sys.stderr.write("Creating config file\n")
    model_details  = read_yaml(args.model_details)
    exemplars_path = os.path.join(output_dir, f"{args.output_prefix}.exemplars.jsonl")
    data_path      = os.path.join(output_dir, f"{args.output_prefix}.data.jsonl")
    prompt_path    = os.path.join(output_dir, f"{args.output_prefix}.prompt.json")
    
    config_content = {
        **model_details,
        "exemplars": {
            "path": exemplars_path,
            "num_exemplars": count_lines(exemplars_path),
            "shuffle": False
        },
        "prompts": {
            "path": prompt_path,
            "prompt_name": args.prompt_name,
            "output_var": args.output_label,
            "labels": [],  # keep this empty for open labels
            "input_vars": [args.input_label]
        },
        "dataset": {
            "dataset_name": args.dataset_name,
            "id_key": args.id_key,
            "path": data_path
        },
        "output": {
            "output_dir": task_output_dir,
            "output_file": f"{args.output_prefix}.output.jsonl",
            "task_name": args.output_prefix
        }
    }

    output_path = os.path.join(output_dir, f"{args.output_prefix}.config.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(config_content, f, sort_keys=False)

    
def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def count_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

if __name__ == "__main__":
    main()


