

#  `text_llm_ui.py` 

This is a simple, non-conversational text-based UI built on [`create_hf_fewshot_task.py`](README.md) and the [`hf-fewshot`](../README.md) package. Assuming those are already in place, to use this UI simply execute
`python text_llm_ui.py` with appropriate arguments, selecting `MODELFILE` to either use the OpenAI API or a local Hugging Face model. 

One of the main intended uses for this UI is to be able to try out prompts interactively using local models, not transmitting data off premises, e.g. for work with potentially sensitive data or to avoid potential contamination of future LLM training data with what you're working on.

This is basically a super low tech version of the Hugging Face Chat UI [[code](https://github.com/huggingface/chat-ui), [UI](https://huggingface.co/chat/)] to use you don't have the time or expertise to [set it up locally](https://github.com/huggingface/chat-ui?tab=readme-ov-file#quickstart), and if the model you want is not among HuggingChat's [available models](https://huggingface.co/chat/models/) or perhaps  you don't want to trust their [privacy policy](https://huggingface.co/chat/privacy/).

## Arguments
```
  --task_creator TASK_CREATOR  Path to create_hf_fewshot_task.py
  --tmp TMP                    Temporary directory (defaults to /tmp)
  --modelfile MODELFILE        Model details file
  -h, --help                   Show this help message and exit
  --verbose                    Enable verbose output
```

Note that if using verbose output, the temporary directory will *not* be deleted automatically, though the verbose output will include information about that directory so you can delete it yourself. A file containing your prompt will be in that directory, so bear this in mind if anything in your prompt is private or sensitive.

## Example

This example uses OpenAI GPT-4 turbo, via the OpenAI API. This means that `OPENAI_API_KEY` must be set in your environment. Also note that it's best to use absolute paths, not relative paths, for file arguments.


```
python text_llm_ui.py \
  --task_creator `pwd`/create_hf_fewshot_task.py \
  --modelfile `pwd`/model_details/model_details_gpt_4000.yaml
```

Here is an illustration of an interaction. Note that the actual call to the LLM can take a little while, e.g. 5-10 seconds is not that unusual.

```
=======================================

PROMPT (terminate with ctl-D, use ctl-C to quit)

Note that each prompt is independent.
This is NOT a conversational interface.
=======================================
Convert these refs to bibtex:

Donahue, T. S. (1984). A Linguist Looks at Tolkien's Elvish. Mythlore, 10(3 (37), 28-31.

Okrand, M. (1992). The Klingon dictionary: the official guide to Klingon words and phrases. Simon and Schuster.


Blanke, D. (2009). Causes of the relative success of Esperanto. Language Problems and Language Planning, 33(3), 251-266.

<USER TYPED: ctl-D>
Creating call to LLM...
=======================================
LLM RESPONSE
=======================================

Here are the BibTeX entries for the references you provided:

```bibtex
@article{Donahue1984,
  author = {Donahue, T. S.},
  title = {A Linguist Looks at Tolkien's Elvish},
  journal = {Mythlore},
  volume = {10},
  number = {3 (37)},
  pages = {28--31},
  year = {1984}
}

@book{Okrand1992,
  author = {Okrand, Marc},
  title = {The Klingon Dictionary: The Official Guide to Klingon Words and Phrases},
  publisher = {Simon and Schuster},
  year = {1992}
}

@article{Blanke2009,
  author = {Blanke, Detlev},
  title = {Causes of the Relative Success of Esperanto},
  journal = {Language Problems and Language Planning},
  volume = {33},
  number = {3},
  pages = {251--266},
  year = {2009}
}

These entries are formatted for use in a BibTeX database, which can be used with LaTeX documents to manage citations and bibliographies. Each entry is given a type (e.g., `@article`, `@book`) and required fields such as author, title, year, and publisher or journal detail.

=======================================

PROMPT (terminate with ctl-D, use ctl-C to quit)

Note that each prompt is independent.
This is NOT a conversational interface.
=======================================
<USER TYPED: ctl-C>
Program terminated by user.
```






