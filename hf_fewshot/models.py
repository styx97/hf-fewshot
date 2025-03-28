import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import os 
from openai import OpenAI 
from abc import ABC, abstractmethod
from typing import List, Dict
import pynvml
from dotenv import load_dotenv

def get_unused_gpu_memory():
    """
    Get the amount of unused GPU memory.
    Returns:
        int: Unused GPU memory in MB.
    """
    pynvml.nvmlInit()
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


def display_gpu_status():
    # Initialize NVML
    pynvml.nvmlInit()
    
    try:
        # Get the number of GPUs in the system
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            print("No GPUs found on this machine.")
            return
        
        # Define headers for the table
        headers = ["GPU ID", "Name", "Memory Usage", "GPU Utilization"]
        header_line = f"{headers[0]:<10} {headers[1]:<20} {headers[2]:<30} {headers[3]:<20}"
        
        # Print table header
        print(header_line)
        print("=" * len(header_line))
        
        for i in range(device_count):
            # Get handle for each GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Retrieve GPU name
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            
            # Retrieve memory information
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = mem_info.total // 1024 ** 2  # Convert bytes to MB
            used_memory = mem_info.used // 1024 ** 2   # Convert bytes to MB
            memory_util_percent = (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
            
            # Retrieve utilization rates
            utilization_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization_rates.gpu
            
            # Construct GPU status line
            memory_usage = f"{used_memory} MB / {total_memory} MB ({memory_util_percent:.2f}%)"
            util = f"{gpu_util:.2f}%"
            line = f"{i:<10} {gpu_name:<20} {memory_usage:<30} {util:<20}"

            # Print each line of GPU status
            print(line)
    
    finally:
        # Make sure NVML is always shutdown
        pynvml.nvmlShutdown()

def labels_vocab_id_map(tokenizer, labels: list[str]) -> dict[str, list[int]]:
    """
    Create a mapping of labels to their tokenized ids.
    """
    label_id_map = {}
    for label in labels: 
        ids = tokenizer.encode(label, add_special_tokens=False)
        label_id_map[label] = ids
    return label_id_map

def get_logsoftmax(x): 
    m = torch.nn.LogSoftmax(dim=1)
    return m(x.view(1, -1))

def get_logsoftmax1(logits_T:torch.FloatTensor, negative:bool=False) -> torch.FloatTensor:
  """
  standard log likelihood for language modeling
  """
  log_probs_T = (logits_T.log_softmax(dim=-1)) ## log probs 

  if negative: ## turn into negative loglikelihood loss
    log_probs_T = -log_probs_T
  return log_probs_T


class FewShotModel(ABC):
    @abstractmethod
    def generate_answer(self, messages: List[Dict]) -> Dict:
        pass

    @abstractmethod
    def generate_answer_batch(self, query_texts: List[str]) -> List[str]:
        pass

class GPTFewShot:
    """
    a class the calls an openai model to generate text
    The openai key is fetched from the env variable OPENAI_API_KEY
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in environment variables")
    

    def __init__(self, 
                 model_name: str,
                 model_details: dict=None,
                 **kwargs # to absorb `labels` and other arguments
                ):
        
        self.model = model_name 
        self.model_details = model_details
        # if no model details are provided, set defaults
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        self.temperature = self.model_details['temperature']
        self.max_tokens = self.model_details['max_new_tokens']
        self.top_p = self.model_details.get("top_p", 1)

        # setting default
        self.label_id_map = {}

    def generate_answer_debug(self, question_text: str):
        return self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "user", "content": question_text}
            ],
            temperature=self.temperature, 
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
    
    def generate_answer(self, messages: list[dict]):
        answer_object = self.client.chat.completions.create(
            model=self.model, 
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

        return answer_object.choices[0].message.content
    
    def generate_answer_batch(self, query_texts: list) -> list[str]: 
        
        answer_texts = []

        for message in query_texts: 
            answer_text = self.generate_answer(message)
            answer_texts.append(answer_text)

        return answer_texts
    
    def generate_answer_batch_scores(self, query_texts: list) -> list[str]: 
        answer_texts = []
        top_logprobs = []
        for message in query_texts: 
            answer_object = self.client.chat.completions.create(
                model=self.model, 
                messages=message,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                logprobs=True,
                top_logprobs=5
                
            )
            answer_text = answer_object.choices[0].message.content
            # This is getting the top n logprobs for the first token!

            label_logprobs = answer_object.choices[0].logprobs.content[0]
            top_logprobs.append({tok.token: tok.logprob for tok in label_logprobs.top_logprobs})
            answer_texts.append(answer_text)
        return answer_texts, top_logprobs
            

class HFFewShot:
    def __init__(self,
                 model_name: str, 
                 model_details: dict=None,
                 labels: list[str]=None,
                ):
        
        """
        A general class for loading huggingface models with a 
        standard set of parameters. 
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        if "quantization" not in model_details or model_details["quantization"] is None: 
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto", 
                torch_dtype="auto"
            ).eval()
            
        else: 
            print("Quantization is set to ", model_details["quantization"])
            # write the quantization logic 
            if model_details["quantization"] == "8bit": 
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,                                                    
                    load_in_8bit=True,
                    device_map="auto"
                ).eval()
                
            elif model_details["quantization"] == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                ).eval()

        # show how much gpu memory is being used per gpu 
        print("Model loaded")
        if self.model.device == "cuda":
            display_gpu_status()
        
        self.max_new_tokens = model_details.get("max_new_tokens", 10)
        self.temperature = model_details.get("temperature", 0.01)
        self.do_sample = model_details.get("do_sample", True)

        if labels and model_details["scores"]:
            self.label_id_map = labels_vocab_id_map(self.tokenizer, labels)
            print("Label ID Map: ", self.label_id_map)
        # TODO: consider setting default for `self.label_id_map` 

    def generate_answer_batch(self, 
                        query_texts: list) -> list[str]: 
    
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        TODO: Get FlashAttention2 to work with this.
        """
        
        messages = [
            self.tokenizer.apply_chat_template(messages, tokenize=False) 
            for messages in query_texts
        ]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_inputs = self.tokenizer(messages, return_tensors="pt", padding=True).to(self.model.device)

        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature, 
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answer_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return [answer_text.split("[/INST]")[-1].strip() for answer_text in answer_texts]


class GemmaFewShot(HFFewShot):
    def __init__(self, 
                 model_name: str, 
                 model_details: dict=None,
                 labels: list[str]=None,
            ):
        
        super().__init__(model_name, model_details, labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate_answer(self, query_text: list[str]) -> str:
        """
        Code to process a single list of conversational history and generate the answer.
        """
        # Applying the chat template to the list of messages (conversational history)
        message = self.tokenizer.apply_chat_template(
            query_text,
            add_generation_prompt=True,
            tokenize=False
        )

        # Tokenizing the message
        model_input = self.tokenizer(
            message,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        # Generating the output
        outputs = self.model.generate(
            **model_input,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decoding the generated text
        answer_text = self.tokenizer.decode(
            outputs[0, model_input['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return answer_text.strip()
        
    def generate_answer_batch(self, query_texts: list) -> list[str]: 
        
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        """
        messages = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            ) for messages in query_texts
        ]

        model_inputs = self.tokenizer(
            messages,
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature, 
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True
        )

        answer_texts = self.tokenizer.batch_decode(outputs[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return answer_texts


class LlamaFewShot(HFFewShot):
    def __init__(self, 
                 model_name: str, 
                 model_details: dict=None,
                 labels: list[str]=None
                ):
        
        super().__init__(model_name, model_details, labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_answer(self, query_text: list[str]) -> str:
        """
        Code to process a single list of conversational history and generate the answer.
        """
        # Applying the chat template to the list of messages (conversational history)
        message = self.tokenizer.apply_chat_template(
            query_text,
            add_generation_prompt=True,
            tokenize=False
        )

        # Tokenizing the message
        model_input = self.tokenizer(
            message,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        # Define terminators for the generation
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Generating the output
        outputs = self.model.generate(
            **model_input,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            eos_token_id=terminators,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decoding the generated text
        answer_text = self.tokenizer.decode(
            outputs[0, model_input['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return answer_text.strip()
        
    def generate_answer_batch_logprobs(self, 
                        query_texts: list) -> list[str]: 
    
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        """
        messages = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            ) for messages in query_texts
        ]

        model_inputs = self.tokenizer(
            messages,
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=self.temperature,
            eos_token_id=terminators,
            return_dict_in_generate=True, 
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True
        )

        answer_texts = self.tokenizer.batch_decode(
            outputs.sequences[:, model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        scores = outputs.scores
        
        #import ipdb; ipdb.set_trace()

        # return the scores and answers
        return {"answers": answer_texts, "scores": scores}


    def generate_answer_batch(self, query_texts: list) -> list[str]: 
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        """
        messages = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            ) for messages in query_texts
        ]

        model_inputs = self.tokenizer(
            messages,
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            eos_token_id=terminators,
            temperature=self.temperature, 
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True
        )

        answer_texts = self.tokenizer.batch_decode(
            outputs[:, model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )


        return answer_texts

class MistralFewShot:
    def __init__(self, 
                model_name: str, 
                labels: list[str]=None, 
                model_details: dict=None):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO: support quantization
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        
        self.max_new_tokens = model_details.get("max_new_tokens", 10)
        self.temperature = model_details.get("temperature", 0.01)
        self.do_sample = model_details.get("do_sample", True)

        if labels and model_details["scores"]:
            self.label_id_map = labels_vocab_id_map(self.tokenizer, labels)
            print("Label ID Map: ", self.label_id_map)      
        # TODO: consider setting default for `self.label_id_map` 

    def debug(self): 
        while True: 
            query_text = input("Enter prompt string: ")
            print(self.generate_answer(query_text))


    def generate_answer(self, question_text: str):
        # This is for debug, mostly
        
        messages = [
                {"role": "user", "content": f"{question_text}"},
            ]

        encoded_text = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encoded_text.to(self.model.device) # has to be in cuda
        outputs = self.model.generate(
            model_inputs, 
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample, 
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # skip_special_tokens doesn't work sometimes so we do it manually
        answer_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return answer_text.split("[/INST]")[-1].strip()
    
    def generate_answer_batch(self, 
                            query_texts: list) -> list[str]: 
        
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        TODO: Get FlashAttention2 to work with this.
        """
        
        messages = [
            self.tokenizer.apply_chat_template(messages, tokenize=False) 
            for messages in query_texts
        ]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        model_inputs = self.tokenizer(messages, return_tensors="pt", padding=True).to(self.model.device)

        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature, 
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answer_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [answer_text.split("[/INST]")[-1].strip() for answer_text in answer_texts]
    
    def generate_answer_batch_scores(self, 
                                     query_texts: list) -> list[str]:

        if not hasattr(self, "labels"):
            raise ValueError("Labels not set. Use set_labels() to set labels.")

        messages = [
            self.tokenizer.apply_chat_template(messages, tokenize=False) 
            for messages in query_texts
        ]
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model_inputs = self.tokenizer(messages, return_tensors="pt", padding=True).to(self.model.device)
        
        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature, 
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        answer_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        # do a sanity check on outputs 

        answer_texts = [answer_text.split("[/INST]")[-1].strip() for answer_text in answer_texts]
        # TODO: Get the logprobs for the labels
        # TODO: need `get_label_logprobs` function
        # stance_logprobs = get_label_logprobs(outputs.scores, self.label_id_map)
        stance_logprobs = None

        return answer_texts, stance_logprobs