import json, yaml, csv
from pathlib import Path

def load_jsonlines(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data

def load_csv(filepath):
    # newline='' is the documented mode for the csv module (quoted embedded
    # newlines parse correctly); utf-8-sig strips a BOM from Excel exports so the
    # first header doesn't become '﻿id'. All values come back as strings,
    # which is what str.format(**record) and json.dumps both want. Empty cells
    # map to '' (DictReader default), which renders cleanly in prompts.
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as f:
        data = [dict(row) for row in csv.DictReader(f)]
    return data

def load_records(filepath, file_type=None):
    """
    Load a dataset/exemplar file as a list of dicts, auto-detecting JSONL vs CSV
    from the file extension. Pass file_type='csv'|'jsonlines' to override.
    """
    if file_type is None:
        ext = Path(filepath).suffix.lower()
        if ext == '.csv':
            file_type = 'csv'
        elif ext in ('.jsonl', '.jl', '.json'):
            file_type = 'jsonlines'
        else:
            raise ValueError(
                f"Cannot infer input format from extension {ext!r} for {filepath}. "
                f"Use a .csv / .jsonl / .jl / .json file, or set dataset.format."
            )

    if file_type == 'csv':
        return load_csv(filepath)
    elif file_type == 'jsonlines':
        return load_jsonlines(filepath)
    else:
        raise ValueError(f"Unsupported file_type: {file_type!r} (expected 'csv' or 'jsonlines')")

# load a yaml file
def load_yaml(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)

def write_jsonlines(data, filepath):
    with open(filepath, 'w') as f:
        for index, line in enumerate(data):
            s = json.dumps(line)
            f.write('\n'*(index>0) + s)

def read_md(filepath: str) -> str:
    with open(filepath, 'r') as f:
        content = f.read()
    return content.strip()


def resolve_text_key(record: dict, text_key: str | None) -> dict:
    """
    Expose the configured text column as the ``text`` prompt variable.

    Datasets and exemplars (CSV columns or JSONL keys both load as dicts) may
    store the text under any field name -- e.g. "Submission text", "body". When
    ``dataset.text_key`` is set, this aliases that field to ``text`` so prompts
    can use a stable ``{text}`` placeholder regardless of the source schema,
    mirroring how ``dataset.id_key`` names the id field.

    Returns a shallow copy with the alias added; a no-op (returns the record
    unchanged) when ``text_key`` is None or already ``"text"``. Raises KeyError
    if the named field is absent so misconfiguration fails loudly.
    """
    if not text_key or text_key == "text":
        return record
    if text_key not in record:
        raise KeyError(
            f"text_key {text_key!r} not found in record with keys {list(record.keys())}"
        )
    return {**record, "text": record[text_key]}


def prep_prompt(targets: dict,
                output_var: str,
                prompt: dict,
                exemplars: list[dict]=None,
                text_key: str | None = None ) -> list:
    """
    Takes a prompt, an exemplar path that is a jsonlines file,
    and targets that contain the data to run predictions on.

    When ``text_key`` is provided, the named field on each target/exemplar is
    aliased to the ``{text}`` placeholder before formatting (see
    ``resolve_text_key``).

    Returns a list of messages that are expected by the
    transformers 'apply_chat_template' method ()
    """
    targets = resolve_text_key(targets, text_key)
    if exemplars:
        exemplars = [resolve_text_key(exemplar, text_key) for exemplar in exemplars]

    beginner_prompt = prompt["zero_shot"]

    if not exemplars:
        return [{
            "role": "user",
            "content":  beginner_prompt.format(**targets)
        }]

    # if exemplars are provided, use the followup prompts (few-shot)
    followup_prompt = prompt["followup"]

    messages = []
    messages.append({
        "role": "user",
        "content": beginner_prompt.format(**exemplars[0])
        })

    messages.append({
        "role": "assistant",
        "content": exemplars[0][output_var]
    })

    for exemplar in exemplars[1:]:
        messages.append({
            "role": "user",
            "content": followup_prompt.format(**exemplar)
        })
        messages.append({
            "role": "assistant",
            "content": exemplar[output_var]
        })

    # finally, pass the target document
    messages.append({
        "role": "user",
        "content": followup_prompt.format(**targets)
    })

    return messages
