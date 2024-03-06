# hf-fewshot
A barebones set of utilities that help run few-shot prompting using hf

Currently, suppoorts few-shot classification (through generation) along with generating logprobs for each class. 

This is meant to be as lightweight as possible, and is not meant to be a full-fledged library.

---

### Files required : 
For running a task, 4 additional files are required. In [example_task](example_task), we have an example of a prediction task, on the [DEBAGREEMENT](https://scale.com/open-av-datasets/oxford) dataset. A brief description of the 4 files is as follows :

- `prompt.json` : Should contain prompts for generation. Each prompt should be in the following format 

    ```python
    {
        "prompt_name": {
            #zero_shot: The initial prompt along with the input variables. In case of zero shot classification, this is the only prompt that will be needed. Example :
            "zero_shot": "You are an expert in US history. Given a question about a US political figurem, answer it with a short paragraph.\n\nQUESTION: {question}\nANSWER: ",

            # followup: A followup prompt that fixes format for exemplars. Example: 
            "followup": "QUESTION: {question}\nANSWER: ",
        }
    }
    ```
    For a more involved example of a prompt, see [examples/sample_prompt.json](example_task/sample_prompt.json). 

- `exemplars.jsonl` : Should contain exemplars in jsonlines format with the input and output variables. 

    ```python
    {"input": "Who was the first president of the United States?","output": "George Washington"}
    ...
    ```

    See [examples/sample_exemplars.jsonl](example_task/sample_exemplars.jsonl) for a sample file.


- `dataset.jsonl`: Should contain the dataset in jsonlines format with the input (and optionally output variables). NOTE: Each dataset item must have a key, to be specified under `dataset` -> `id_key` in `config.json`

    ```python
    {"input": "Who was the first president of the United States?","output": "George Washington"}
    ...
    ```

    See [examples/sample_dataset.jsonl](example_task/sample_dataset.jsonl) for a sample file.

- `config.json`: Should contain various hyperparameters and data paths in a yaml parsable format. See [configs/example_config.json](configs/example_config.yml) for a sample file.