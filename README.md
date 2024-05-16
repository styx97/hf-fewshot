# hf-fewshot
A barebones set of utilities that help run few-shot prompting using hf

Currently, suppoorts few-shot classification (through generation) along with generating logprobs for each class. 

This is meant to be as lightweight as possible, and is not meant to be a full-fledged library.

---

#### Files required : 
For running a task, 4 additional files are required. [example_task](example_task) contains the setup for running a simple question answering task. A brief description of the 4 files is as follows :

- A `prompt.json` should contain prompts for generation. Each prompt should be in the following format 

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

- An `exemplars.jsonl` should contain exemplars in jsonlines format with the input and output variables. 

    ```python
    {"input": "Who was the first president of the United States?","output": "George Washington"}
    ...
    ```

    See [examples/sample_exemplars.jsonl](example_task/sample_exemplars.jsonl) for a sample file.


- A `dataset.jsonl` should contain the dataset in jsonlines format with the input (and optionally output variables). NOTE: Each dataset item must have a key, to be specified under `dataset` -> `id_key` in `config.json`

    ```python
    {"q_id": "1001", "input": "Who was the second president of the United States?", "output": "John Adams"}
    ...
    ```

    See [examples/sample_dataset.jsonl](example_task/sample_dataset.jsonl) for a sample file.

- A `config.yml` file should contain various hyperparameters and data paths in a yaml parsable format. See [configs/example_config.yml](configs/example_config.yml) for a sample file.



---
#### Installation :

Create an empty conda enironment: 

```bash
conda create -n hf-fewshot python=3.11
conda activate hf-fewshot
```

After cloning the repository, run the following commands: 

```bash
# first create an empty conda environment 
conda create -n hf_fewshot python=3.11

cd hf-fewshot
pip install -e .
```

- If using OpenAI, add the OpenAI API key to your environment variables as `OPENAI_API_KEY`. If using conda, you can do this by running `conda env config vars set OPENAI_API_KEY=<your key>`.

- If using a huggingface "Gated" model (such as Llama 3), add the Huggingface access token to an environment variable called `HF_TOKEN` by running `conda env config vars set HF_TOKEN=<your token>`.


#### Running a task :

To run fewshot prompting, run the following command : 

```bash
hf_fewshot --config configs/example_config.yml
```

If `output_file` is specified in the config, it will be saved to that location. Otherwise, a file will be automatically generated in that directory prior to saving. 