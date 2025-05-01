# hf-pairwise
A barebones set of utilities to perform pairwise comparisons to estimate scores of text items along a particular dimension. 

### What this package does

This package provides a simple interface to perform pairwise comparisons of text items using a comparison construct. It: 

- Creates pairwise matches between items in a dataset.

- Performs the pairwise comparisons using an LLM (called from OpenAI, Huggingface, etc.) using a prompt template.

- Runs Bradley-Terry (1952) model to estimate scores for each item based on the pairwise comparisons.


### Required Files
For running a task, you need: 

- A prompt folder that has a prompt file titled `zero_shot.md` (for zero-shot prompting) and an additional file called `followup.md` (for followup prompting using few-shot exemplars).

- A dataset of text items to be scored, in jsonlines format. Each item should have an `id` key and a `text` key. 

- A config file in yaml format that specifies the paths to the prompt, dataset, and other hyperparameters. See [example_pairwise_task/sample_pairwise_config.yaml](sample_pairwise_config) for a sample file, and [example_pairwise_task](example pairwise task) for a complete example. 

### Output Files

Intermediate and final output files are generated in the `output` folder specified in the config file.

Intermediate files include:
- the pairwise dataset. 
- the pairwise comparisons made by the LLM.

The final output file is a jsonlines file with the estimated scores for each item in the dataset, along with their ids.

### How to run

```python hf_pairwise/get_bt_scores.py --config example_pairwise_task/sample_pairwise_config.yaml```