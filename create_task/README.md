

#  `create_hf_fewshot_task.py` 

This program takes inputs for a few-shot task in a convenient format (CSV and text), and automatically produces a directory containing the JSON and YAML files needed by the hf-fewshot package, making it unnecessary to create the files directly in those formats. When done it tells you the command line you need to execute.

**For a simple, non-conversational text-based UI based on this, see [TextUI.md](TextUI.md).**

The `hf-fewshot` package requires four files for any given task:

- `prompt.json`, which contains the LLM prompt
- `exemplars.jsonl`, which contains exemplars of input/output for few-shot learning
- `dataset.jsonl`, which contains the test items
- `config.yml`, a configuration file specifying the above files, model details, and where to put output

Below find:

- Explanation of program arguments

- A copy-pastable example of running the program using the OpenAI API

- Information on using HuggingFace models locally 

## Explanation of program arguments
```
# Arguments related to the output of this program
  --output_dir		Directory for hf_fewshot task files created by this program, e.g. the config file
  --output_prefix	Prefix for hf_fewshot task filenames

# Arguments identifying names/labels for data 
  --id_key ID_KEY   Column name for IDs (defaults to 'ID')
  --input_label		Column name for input items (defaults to 'question')
  --output_label	Column name for output items (defaults to 'answer')
  --output_label_options  Specifies allowed answer options as (i) list of string or integer values or (ii) a string specifying a range like <lowest>-<highest> (defaults to None, open answer)

# Info about the specific language model that will be used
  --model_details	YAML file with model details

# Arguments related to creation of the prompt
  --prompt_name		Name of the prompt (defaults to 'my_prompt')
  --instruction		Text file containing instruction part of the prompt
  --begin			String delimiting start of output (defaults to '[begin]')
  --end				String delimiting end of output (defaults to '[end']
  --followup		String for followup to instruction (should contain the begin/end delimiters in outputs)

# Arguments related to "training" (exemplars) and test 
  --exemplars		CSV file with exemplars (id_key, input_label, output_label)
  --dataset_name	Name of the dataset
  --dataset			CSV file with test items dataset (id_key, input_label, output_label); outputs optional
  --task_output_dir	Directory where LLM task output should go when hf_fewshot is called with this config
```

Notice that the `--followup` argument is a string that will become part of the prompt. Make sure the labels for input and output are used there (e.g. 'question', 'answer').

The begin/end answer delimiters can be used to ensure that answers can easily be parsed from LLM output. You can disable them by providing `''` as the value for each. If they are provided, then the program will check to make sure that they are present in the `followup` string and in the outputs in the CSV files.

## Running an example

You should execute this example in the directory containing `create_hf_fewshot_task.py`. A few notes:

- If `output_dir` is not a full directory path, it will be created relative to the current directory where you execute the python.

- Same for directory `qa_example_task_output`.

- In this example the value of `''` for begin/end delimiters is used to override the default values of `[begin]` and `[end]`. Notice, correspondingly, that in `followup` the `{answer}` has no delimiters around it.

- This example uses OpenAI GPT-4 turbo, via the OpenAI API. This means that `OPENAI_API_KEY` must be set in your environment. 


```
python create_hf_fewshot_task.py \
  --output_dir                   qa_example_hf_files \
  --output_prefix                qa_out \
  --id_key                       ID \
  --input_label                  question \
  --output_label                 answer \
  --output_label_options         options \
  --model_details                model_details/model_details_gpt.yaml \
  --prompt_name                  qa_prompt \
  --instruction                  qa_example_task_data/instruction.txt \
  --begin                        '' \
  --end                          '' \
  --followup                     "Question: {question}\nAnswer: {answer}" \
  --exemplars                    qa_example_task_data/sample_exemplars.csv \
  --dataset_name                 example_qa_dataset \
  --dataset                      qa_example_task_data/sample_dataset.csv \
  --task_output_dir              qa_example_task_output 
```



## Contents of the model_details YAML file

### GPT (which runs using the OpenAI API)

Here are the contents of `model_details/model_details_gpt.yaml`. Note, in particular, the low value for `max_new_tokens`, which limits the length of the LLM response. This particular set of model details is set up for classification under the assumption that the LLM response will contain short labels. Set that value higher if/as necessary.

```
model_details:
    model_name: gpt-4-turbo
    model_family: gpt
    scores: False
    batch_size: 8
    max_new_tokens: 10
    temperature: 0.01
```

### HuggingFace models (which run locally)

We assume you have HuggingFace models installed locally. (See section below on relevant environment variables.) 

The comments are some practical notes. With regard to temperature, note that the hf-fewshot package fixes top_p at 1. See the discussion of top-p (nucleus) sampling [here](https://huggingface.co/blog/how-to-generate) for discussion.


```
model_name: NAME_OF_HUGGINGFACE_MODEL
    model_family: hf-general
       # Use this for huggingface models
    scores: False
       # don't change for now
    batch_size: 8
       # Depends on GPU memory after model is loaded, and size of prompts
       # 8 is a good start, change to 16 if you don't run out of GPU memory; if so, possibly truncate inputs
       # if data are inconsistent in length, some batches might run out of memory and some not
    max_new_tokens: 32
       maximum tokens in LLM loutput - generally good to keep as short as possible
    temperature: 0.7
       # Note that temperature affects different models differently.
       # Typically low for classification, IE, etc.; high for generation creativity/variety
```
       

### Environment variables for using HuggingFace models

#### bash
```
# Set environment variables if you are using HuggingFace models locally
export SCRATCH_DIR="PARTITION_CONTAINING_MODELS"
export CACHE_DIR="${SCRATCH_DIR}/huggingface_cache"
export XDG_CACHE="${CACHE_DIR}/.cache"
export HF_HOME="${XDG_CACHE}/huggingface"
export TRANSFORMERS_CACHE="${XDG_CACHE}/huggingface"

# TO DO: These are probably not required -- remove?
export PYSERINI_CACHE="${CACHE_DIR}/pyserini"
export FLAIR_CACHE_ROOT="${XDG_CACHE}/flair"
export PIP_CACHE_DIR="${CACHE_DIR}/.pip"
export FLAIR_CACHE_ROOT="${XDG_CACHE}/flair"

```

#### Old-school: tcsh

```
# Make the hf-fewshot executable available in your path
setenv PATH                YOUR_PATH_TO_LOCAL_CONDA/envs/llm/bin/:$PATH

# Set environment variables if you are using HuggingFace models locally
setenv SCRATCH_DIR         PARTITION_CONTAINING_MODELS
setenv CACHE_DIR           "${SCRATCH_DIR}/huggingface_cache"
setenv XDG_CACHE           "${CACHE_DIR}/.cache"
setenv HF_HOME             "${XDG_CACHE}/huggingface"
setenv TRANSFORMERS_CACHE  "${XDG_CACHE}/huggingface"

# TO DO: These are probably not required -- remove?
setenv PYSERINI_CACHE      "${CACHE_DIR}/pyserini"
setenv FLAIR_CACHE_ROOT    "${XDG_CACHE}/flair"
setenv PIP_CACHE_DIR       "${CACHE_DIR}/.pip"
setenv FLAIR_CACHE_ROOT    "${XDG_CACHE}/flair"
```



****