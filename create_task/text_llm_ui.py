
########################################################################################################
# Written by Philip using ChatGPT 4o on July 5, 2024.
# Minor edits to the results of the prompt below
#
# 
# You are an expert programmer who writes clear, well documented python code.
# You are writing a text-based interface.
# 
# The program should take an optional commandline argument --task_creator that defaults to ./create_hf_fewshot_task.py
# The program should also take an optional commandline argument --tmp that defaults to /tmp.
# Also an optional commandline argument --modelfile that defaults to ./model_details/model_details_gpt_4000.yaml
# Also include a boolean flag --verbose that defaults to False
# 
# Within an infinite loop, the program should do the following. If --verbose is true, then include a report
# to stdout every time you create a directory, create or delete a file or directory, or execute a command.
# 
#   Create a 10-character random string dirname, set tempdir to <tmp>/<dirname>, and create that directory. 
# 
#   On stdout, write: ============= newline PROMPT newline
# 
#   Wait for the user's input on stdin. Input whatever the user copy/pastes, terminated 
#   by control-d, as the variable 'prompt'.
# 
#   Create directory <tempdir>/taskdata
# 
#   In <tempdir>/taskdata, create an empty file called empty_exemplars.csv
# 
#   In <tempdir>/taskdata, create a file called dummy_dataset.csv that contains the following:
# 
#     ID,question,answer
#     0,"Please follow the instructions above.",""
# 
#   In <tempdir>/taskdata, create a file called instruction.txt that contains <prompt> followed by a newline
# 
#   Create a string cmd containing the following command:
# 
#     python <task_creator> 
#       --output_dir                   <tempdir>/hf_files 
#       --output_prefix                text_ui_out 
#       --id_key                       ID 
#       --input_label                  question 
#       --output_label                 answer 
#       --model_details                <modelfile> 
#       --prompt_name                  text_prompt 
#       --instruction                  <tempdir>/task_data/instruction.txt 
#       --begin                        '' 
#       --end                          '' 
#       --followup                     "" 
#       --exemplars                    <tempdir>/task_data/empty_exemplars.csv 
#       --dataset_name                 prompt_as_dataset 
#       --dataset                      <tempdir>/task_data/dummy_dataset.csv 
#       --task_output_dir              <tempdir>/task_output 
# 
#   Execute the command cmd. If it fails with an error, provide an informative error message, 
#   e.g. the reason it failed, and exit.
# 
#   If the command succeeded:
# 
#     On stdout, write:  ============= newline LLM RESPONSE newline ============= newline
# 
#     Read jsonl file <tempdir>/task_output/text_ui_out.output.jsonl
# 
#     If verbose, report the contents prettyprinted to stdout followed by a newline.
# 
#     Print the value of the jsonl 'response' element on stdout. If there is an error, e.g. 
#     no response element exists, badly formed jsonl, etc., report that there was an error informatively
# 
#     If verbose is not set, remove <tempdir> 
#     Otherwise report that you are not removing it
# 
#     Return to the top of the loop
# 
# The program should terminate if the user types control-c.
########################################################################################################     
import argparse
import os
import tempfile
import shutil
import subprocess
import random
import string
import sys
import json

def create_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def main():
    parser = argparse.ArgumentParser(description="Text-based interface for creating few-shot tasks.")
    parser.add_argument("--task_creator", default="./create_hf_fewshot_task.py", help="Path to the task creator script")
    parser.add_argument("--tmp", default="/tmp", help="Temporary directory (defaults to /tmp)")
    parser.add_argument("--modelfile", default="./model_details/model_details_gpt_4000.yaml", help="Model details file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    try:
        while True:
            dirname = create_random_string()
            tempdir = os.path.join(args.tmp, dirname)
            os.makedirs(tempdir)
            if args.verbose:
                print(f"Created directory: {tempdir}")

            print("=======================================\n\nPROMPT (terminate with ctl-D, use ctl-C to quit)\n")
            print("Note that each prompt is independent.")
            print("This is NOT a conversational interface.")
            print("=======================================")
            prompt = sys.stdin.read().strip()

            taskdata_dir = os.path.join(tempdir, "taskdata")
            os.makedirs(taskdata_dir)
            if args.verbose:
                print(f"Created directory: {taskdata_dir}")

            empty_exemplars_path = os.path.join(taskdata_dir, "empty_exemplars.csv")
            with open(empty_exemplars_path, "w") as f:
                pass
            if args.verbose:
                print(f"Created file: {empty_exemplars_path}")

            dummy_dataset_path = os.path.join(taskdata_dir, "dummy_dataset.csv")
            with open(dummy_dataset_path, "w") as f:
                f.write('ID,question,answer\n0,"Please follow the instructions above.",""\n')
            if args.verbose:
                print(f"Created file: {dummy_dataset_path}")

            instruction_path = os.path.join(taskdata_dir, "instruction.txt")
            with open(instruction_path, "w") as f:
                f.write(f"{prompt}\n")
            if args.verbose:
                print(f"Created file: {instruction_path}")

            cmd = (
                f"python {args.task_creator} "
                f"--output_dir {tempdir}/hf_files "
                f"--output_prefix text_ui_out "
                f"--id_key ID "
                f"--input_label question "
                f"--output_label answer "
                f"--model_details {args.modelfile} "
                f"--prompt_name text_prompt "
                f"--instruction {instruction_path} "
                f"--begin '' "
                f"--end '' "
                f"--followup \"\" "
                f"--exemplars {empty_exemplars_path} "
                f"--dataset_name prompt_as_dataset "
                f"--dataset {dummy_dataset_path} "
                f"--task_output_dir {tempdir}/task_output"
            )


            if args.verbose:
                command_path = os.path.join(tempdir, "command.txt")
                with open(command_path, "w") as f:
                    pretty_cmd = (
                        f"python {args.task_creator} \\\n"
                        f"  --output_dir {tempdir}/hf_files \\\n"
                        f"  --output_prefix text_ui_out \\\n"
                        f"  --id_key ID \\\n"
                        f"  --input_label question \\\n"
                        f"  --output_label answer \\\n"
                        f"  --model_details {args.modelfile} \\\n"
                        f"  --prompt_name text_prompt \\\n"
                        f"  --instruction {tempdir}/taskdata/instruction.txt \\\n"
                        f"  --begin '' \\\n"
                        f"  --end '' \\\n"
                        f"  --followup \"\" \\\n"
                        f"  --exemplars {tempdir}/taskdata/empty_exemplars.csv \\\n"
                        f"  --dataset_name prompt_as_dataset \\\n"
                        f"  --dataset {tempdir}/taskdata/dummy_dataset.csv \\\n"
                        f"  --task_output_dir {tempdir}/task_output\n"
                    )
                    f.write(pretty_cmd)
                if args.verbose:
                    print(f"Created file: {command_path}")

            print("Creating call to LLM...")
            if args.verbose:
                print(f"Created file: {command_path}")
                print(f"Running: {cmd}")
                
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Command failed with error: {result.stderr}")
                break

            hf_fewshot_cmd = f"hf_fewshot --config {tempdir}/hf_files/text_ui_out.config.yaml"
            if args.verbose:
                print(f"Running: {hf_fewshot_cmd}")
                
            result = subprocess.run(hf_fewshot_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Config command failed with error: {result.stderr}")
                break

            print("=======================================\nLLM RESPONSE\n=======================================\n")
            output_file = os.path.join(tempdir, "task_output", "text_ui_out.output.jsonl")
            try:
                with open(output_file, "r") as f:
                    for line in f:
                        response_json = json.loads(line)
                        if args.verbose:
                            print(json.dumps(response_json, indent=4))
                        if 'response' in response_json:
                            print(response_json['response'])
                        else:
                            print("Error: 'response' element not found in the output.")
            except FileNotFoundError:
                print("Error: Output file not found.")
            except json.JSONDecodeError:
                print("Error: Failed to parse the output JSON.")

            if not args.verbose:
                shutil.rmtree(tempdir)
            else:
                print(f"Not removing directory: {tempdir}")

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")

if __name__ == "__main__":
    main()
