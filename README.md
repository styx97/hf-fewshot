# hf-fewshot
A barebones set of utilities that help run few-shot prompting using hf

Currently, supports few-shot classification (through generation) along with generating logprobs for each class. 

This is meant to be as lightweight as possible, and is not meant to be a full-fledged library.

---

#### Files required : 

For running a task, five additional files are required. [example_tasks](example_tasks) contains the setup for three tasks: Question answering, pairwise comparison, and scoring.

A brief description of the five required files based on [Question answering](example_tasks/question_answering/) is as follows :


- A `config.yml` file should contain various hyperparameters and data paths in a yaml parsable format. See [configs/example_qna_config.yml](configs/example_qna_config.yml) for a sample file.

- A `zero_shot.md` should contain the prompt for generation.

    ```markdown
    You are an expert in US history. Given a question about a US political figurem, answer it with a short paragraph.
    
    QUESTION: {question}
    ````
- A `followup.md` should contain the format for few-shot examplars (optional for few-shot in-context learning):
    ```markdown
    QUESTION: {question}
    ANSWER: 
    ```
- A `dataset.jsonl` should contain the dataset in jsonlines format with the input (and optionally output variables). NOTE: Each dataset item must have a key, to be specified under `dataset` -> `id_key` in `config.json`

    ```python
    {"q_id": "1001", "input": "Who was the second president of the United States?", "output": "John Adams"}
    ...
    ```

    See [examples/sample_dataset.jsonl](example_task/sample_dataset.jsonl) for a sample file.
- An `exemplars.jsonl` should contain exemplars in jsonlines format with the input and output variables. 

    ```python
    {"input": "Who was the first president of the United States?","output": "George Washington"}
    ...
    ```

    See [examples/sample_exemplars.jsonl](example_task/sample_exemplars.jsonl) for a sample file.


See the [create_task](./create_task) subdirectory for convenient code that will create these files automatically for you from CSV and text files, which you may find more convenient.

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
hf_fewshot --config configs/example_qna_config.yml
```

If `output_file` is specified in the config, it will be saved to that location. Otherwise, a file will be automatically generated in that directory prior to saving.




