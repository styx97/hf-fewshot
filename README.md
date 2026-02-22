# hf-fewshot
A barebones set of utilities to perform pairwise comparisons to estimate scores of text items along a particular dimension. 

## What this package does

This package provides a simple interface to perform zero/few-shot inference using LLMs hosted on huggingface. Features include quantization, dynamic batching, etc. 

This package also contains code accompanying the paper [PairScale](https://aclanthology.org/2025.findings-naacl.94/), including end-to-end code on generating scores from pairwise comparisons. 

## macOS Compatibility Fixes 

### Summary

The current branch in the hf-fewshot repository, hf-fewshot-mac, includes code changes made to enable the hf-fewshot package to run successfully on macOS systems, particularly Apple Silicon (arm64) Macs. The changes address several issues that prevented the package from functioning on macOS environments lacking NVIDIA GPU drivers.  These changes were made with the assistance of Claude Code.

### Environment Context

**Testing Environment:**

- Platform: macOS (Darwin 23.4.0)
- Architecture: arm64 (Apple Silicon)
- Python: 3.13 
- RAM: 32GB

**Installation Context:**
The package had initial installation challenges on macOS due to:

- autoawq build failures requiring torch to be pre-installed
- NVIDIA-specific dependencies that fail gracefully on systems without NVIDIA hardware

### Code Changes Overview

All changes were made to maintain backward compatibility while adding robust error handling for non-NVIDIA systems. The fixes enable the package to work seamlessly on macOS while preserving full functionality on systems with NVIDIA GPUs.

---

### Fix 1: GPU Status Handling for Non-NVIDIA Systems

#### Problem
The original code assumed NVIDIA GPU drivers were available and would crash with `pynvml.NVMLError` on macOS systems:

```python
def get_unused_gpu_memory():
    pynvml.nvmlInit()  # Crashes on macOS without NVIDIA drivers
    device_count = pynvml.nvmlDeviceGetCount()
    # ... rest of function

```

```python 
def display_gpu_status():
    pynvml.nvmlInit()  # Crashes on macOS without NVIDIA drivers
    # ... rest of function

```

**Error encountered:**

```
pynvml.NVMLError: NVML Shared Library Not Found

```

#### Solution
Added comprehensive error handling with try/catch blocks to gracefully handle missing NVIDIA drivers:

**File:** `hf_fewshot/models.py`

**Lines 12-40 - Modified `get_unused_gpu_memory()`:**

```python
def get_unused_gpu_memory():
    """
    Get the amount of unused GPU memory.
    Returns:
        int: Unused GPU memory in MB.
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return 0
    
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        total_unused_memory = 0
        total_available_memory  = 0
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = info.total // 1024 ** 2  # Convert bytes to MB
            used_memory = info.used // 1024 ** 2   # Convert bytes to MB
            unused_memory = total_memory - used_memory
            total_unused_memory += unused_memory
            total_available_memory += total_memory
            
        pynvml.nvmlShutdown()
        return round((total_unused_memory / total_available_memory) * 100, 5)
    except pynvml.NVMLError:
        return 0

```

**Lines 43-60 - Modified `display_gpu_status()`:**

```python
def display_gpu_status():
    # Initialize NVML
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        print("No NVIDIA GPUs found on this machine.")
        return
    
    try:
        # Get the number of GPUs in the system
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            print("No GPUs found on this machine.")
            return
        # ... rest of existing GPU enumeration logic
    except pynvml.NVMLError:
        print("Error accessing GPU information.")

```

#### Impact

- Functions now return gracefully (0 for memory, informative message for status) on non-NVIDIA systems
- NVIDIA GPU systems retain full functionality
- No changes needed to calling code

---

### Fix 2: Missing Method Implementation for Batch Processing with Logprobs

#### Problem
The `GPTFewShot` class was missing the `generate_answer_batch_logprobs()` method that was being called by the classification system, causing runtime errors:

```
AttributeError: 'GPTFewShot' object has no attribute 'generate_answer_batch_logprobs'

```

The `HFFewShot` class had this method, but `GPTFewShot` (for OpenAI models) did not implement it, creating an inconsistent interface.

#### Solution
Added the missing method to `GPTFewShot` class to match the interface expected by the classification system.

**File:** `hf_fewshot/models.py`

**Lines 204-228 - Added `generate_answer_batch_logprobs()` to `GPTFewShot`:**

```python
def generate_answer_batch_logprobs(self, message_objects: List[Dict]) -> Mapping[str, Union[List[str], Tuple]]:
    """
    Generate answers with logprobs for batch processing.
    Returns dictionary with 'answers' and 'scores' keys to match HFFewShot interface.
    """
    answer_texts = []
    top_logprobs = []
    
    for message in message_objects:
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            logprobs=True,
            top_logprobs=20
        )
        answer_text = response.choices[0].message.content
        answer_texts.append(answer_text)

        label_logprobs = response.choices[0].logprobs.content[0]
        top_logprobs.append({tok.token: tok.logprob for tok in label_logprobs.top_logprobs})
    
    return {"answers": answer_texts, "scores": tuple(top_logprobs)}

```

**Lines 298-326 - Also added to `HFFewShot` class (was referenced but missing):**

```python
def generate_answer_batch_logprobs(self, message_objects: List[Dict]) -> Mapping[str, Union[List[str], Tuple[torch.FloatTensor]]]:
    """
    Code to batch process multiple questions with logprobs.
    """
    messages = [
        self.tokenizer.apply_chat_template(messages, tokenize=False) 
        for messages in message_objects
    ]

    self.tokenizer.pad_token = self.tokenizer.eos_token
    
    model_inputs = self.tokenizer(messages, return_tensors="pt", padding=True).to(self.model.device)

    outputs = self.model.generate(
        **model_inputs, 
        max_new_tokens=self.max_new_tokens,
        do_sample=self.do_sample,
        temperature=self.temperature if self.do_sample else None,
        pad_token_id=self.tokenizer.eos_token_id,
        return_dict_in_generate=True, 
        output_scores=True
    )

    answer_texts = self.tokenizer.batch_decode(
        outputs.sequences[:, model_inputs.input_ids.shape[-1]:], 
        skip_special_tokens=True
    )

    return {"answers": answer_texts, "scores": outputs.scores}

```

#### Impact

- OpenAI models can now be used for classification tasks requiring logprobs
- Consistent interface across all model classes
- Enables full functionality with GPT models

---

### Fix 3: Tokenizer Padding Configuration for Decoder-Only Models

#### Problem
The original tokenizer initialization caused warnings when using decoder-only models like GPT-2 and Gemma:

```
UserWarning: Padding side was not set to 'left', which is required for decoder-only models

```

#### Solution
Updated tokenizer initialization to set proper padding side for decoder-only models.

**File:** `hf_fewshot/models.py`

**Line 242 - Modified `HFFewShot.__init__()`:**

```python
## Before:
self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

## After:
self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')

```

#### Impact

- Eliminates padding warnings for decoder-only models
- Ensures correct token alignment during batch processing
- Improves generation quality for models that require left padding

---

### Fix 4: Improved Classification Logic for Non-Label Scenarios

#### Problem
The classification code always attempted to call `generate_answer_batch_logprobs()` and process scores, even when no labels were defined and scores weren't needed:

```python
## Original problematic code
batched_output = model.generate_answer_batch_logprobs(batch_query_texts)
logprobs = get_logprobs(batched_output["scores"])
preferences = get_option_preferences(model, logprobs, list(model.label_id_map.keys())) if has_labels else [None] * len(batch_query_texts)

```

This caused issues with OpenAI models when `scores: False` was set in the configuration.

#### Solution
Added conditional logic to only process logprobs when labels are actually present.

**File:** `hf_fewshot/classifiers.py`

**Lines 275-283 - Modified inference logic in `run_inference()`:**

```python
## Before: Always called generate_answer_batch_logprobs
batched_output = model.generate_answer_batch_logprobs(batch_query_texts)
logprobs = get_logprobs(batched_output["scores"])
preferences = get_option_preferences(model, logprobs, list(model.label_id_map.keys())) if has_labels else [None] * len(batch_query_texts)

## After: Conditional logic based on label presence
if has_labels:
    batched_output = model.generate_answer_batch_logprobs(batch_query_texts)
    logprobs = get_logprobs(batched_output["scores"])
    preferences = get_option_preferences(model, logprobs, list(model.label_id_map.keys()))
    answers = batched_output["answers"]
else:
    answers = model.generate_answer_batch(batch_query_texts)
    preferences = [None] * len(batch_query_texts)

```

**Line 287 - Fixed variable reference:**

```python
## Before:
for item_id, preference, answer in zip(ids, preferences, batched_output["answers"]):

## After:
for item_id, preference, answer in zip(ids, preferences, answers):

```

#### Impact

- Eliminates unnecessary logprob processing for simple Q&A tasks
- Prevents data type errors when OpenAI models return different score formats
- Improves performance by avoiding unnecessary API calls with logprobs enabled

---

### Testing Results

**Verification Process:**

1. **Package Import Test:** `python -c "import hf_fewshot; print('Package imported successfully')"`
2. **Console Script Test:** `hf_fewshot --help` 
3. **Local Model Test:** `hf_fewshot --config configs/test_local_config.yml`
4. **OpenAI Model Test:** `hf_fewshot --config configs/test_local_config.yml` (with GPT-4o-mini)

**Results:**

- ✅ All 14 questions processed successfully
- ✅ Output file generated correctly: `outputs/test_local_output.jsonl`
- ✅ No crashes or errors during execution
- ✅ Proper answers generated for Q&A dataset

**Sample Output:**

```json
{"q_id": 0, "output": "Rishi Sunak"}
{"q_id": 1, "output": "Mount Everest"}
{"q_id": 2, "output": "China"}
{"q_id": 4, "output": "Paris"}
{"q_id": 5, "output": "Mars"}

```

---

### Backward Compatibility

All changes maintain full backward compatibility:

- NVIDIA GPU systems retain full GPU monitoring functionality
- All existing model classes continue to work unchanged  
- Configuration files require no modifications
- API interfaces remain consistent

### Performance Impact

The changes have minimal performance impact:

- GPU functions return immediately (0 or message) on non-NVIDIA systems
- Classification logic is more efficient, avoiding unnecessary logprob processing
- Tokenizer improvements reduce warnings and improve generation quality

### Recommendations for Integration

1. **Consider upstreaming these fixes** - They solve common issues on macOS without breaking existing functionality
2. **Test on additional macOS versions** - These fixes were tested on macOS 14.4 with Apple Silicon
3. **Documentation updates** - Update installation docs to mention macOS compatibility
4. **CI/CD considerations** - Consider adding macOS to your testing matrix

### Installation Notes for macOS Users

The recommended installation process that works with these fixes:

```bash
pip install 'torch>=2.3.0'  # Install torch first
pip install transformers accelerate bitsandbytes sentencepiece pynvml python-dotenv openai optimum
pip install -e . --no-deps  # Skip problematic dependencies like autoawq

```




