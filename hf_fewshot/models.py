import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import BitsAndBytesConfig
import os 
from openai import OpenAI 
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple, Mapping
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

def labels_vocab_id_map(tokenizer, labels: List[str]) -> Mapping[str, List[int]]:
    """
    Create a mapping of labels to their tokenized ids.
    """
    # if the labels are not a list of strings, make it so 
    labels = map(lambda x: str(x).strip(), labels) # ensure all labels are strings and stripped of whitespace

    label_id_map = {}
    for label in labels: 
        ids = tokenizer.encode(label, add_special_tokens=False)
        label_id_map[label] = ids
    return label_id_map

def get_logsoftmax(x): 
    m = torch.nn.LogSoftmax(dim=1)
    return m(x.view(1, -1))

def get_logsoftmax1(logits_T: torch.FloatTensor, negative: bool=False) -> torch.FloatTensor:
  """
  standard log likelihood for language modeling
  """
  log_probs_T = (logits_T.log_softmax(dim=-1)) ## log probs 

  if negative: ## turn into negative loglikelihood loss
    log_probs_T = -log_probs_T
  return log_probs_T


class FewShotModel(ABC):
    @abstractmethod
    def generate_answer(self, messages: List[Dict]) -> str:
        pass

    @abstractmethod
    def generate_answer_batch(self, message_objects: List[Dict]) -> list[str]:
        pass

class GPTFewShot:
    """
    a class the calls an openai model to generate text
    The openai key is fetched from the env variable OPENAI_API_KEY
    """

    def __init__(self, 
                 model_name: str,
                 model_details: dict=None,
                 **kwargs # to absorb `labels` and other arguments
                ):
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        
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
    
    def generate_answer(self, messages: List[Dict]):
        answer_object = self.client.chat.completions.create(
            model=self.model, 
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

        return answer_object.choices[0].message.content
    
    def generate_answer_batch(self, message_objects: List[Dict]) -> list[str]:
        
        answer_texts = []

        for message in message_objects:
            answer_text = self.generate_answer(message)
            answer_texts.append(answer_text)
        
        return answer_texts
    
    def generate_answer_batch_scores(self, message_objects: List[Dict]) -> Mapping[str, Union[List[str], Mapping[str, float]]]:
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
            # NOTE: not compatible with other model's logprobs output
            top_logprobs.append({tok.token: tok.logprob for tok in label_logprobs.top_logprobs})
        
        return answer_texts, top_logprobs
            
class HFFewShot:
    def __init__(self,
                 model_name: str, 
                 model_details: dict=None,
                 labels: List[str]=None,
                 model_class=AutoModelForCausalLM,
                ):
        """
        A general class for loading huggingface models with a 
        standard set of parameters. 
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if model_details is None:
            model_details = {}
        
        if "quantization" not in model_details or model_details["quantization"] is None: 
            self.model = model_class.from_pretrained(
                model_name, 
                device_map="auto", 
                torch_dtype="auto"
            ).eval()
            
        else: 
            print("Quantization is set to ", model_details["quantization"])
            # write the quantization logic 
            if model_details["quantization"] == "8bit": 
                self.model = model_class.from_pretrained(
                    model_name,                                                    
                    load_in_8bit=True,
                    device_map="auto",
                    torch_dtype="auto"
                ).eval()
                
            elif model_details["quantization"] == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )

                self.model = model_class.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype="auto"
                ).eval()

        # show how much gpu memory is being used per gpu 
        print("Model loaded")
        if self.model.device == "cuda":
            display_gpu_status()
        
        self.max_new_tokens = model_details.get("max_new_tokens", 10)
        self.temperature = model_details.get("temperature", 0.01)
        self.do_sample = model_details.get("do_sample", True)

        if labels and "scores" in model_details and model_details["scores"]:
            
            self.label_id_map = labels_vocab_id_map(self.tokenizer, labels)
            print("Label ID Map: ", self.label_id_map)
        
        # TODO: consider setting default for `self.label_id_map`

    def generate_answer_batch(self, message_objects: List[Dict]) -> List[str]:
    
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        TODO: Get FlashAttention2 to work with this.
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
        )

        answer_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return [answer_text.split("[/INST]")[-1].strip() for answer_text in answer_texts]
    
    @staticmethod
    def _prompt_to_messages(
        prompt: str, 
        system_message: Optional[str]=None
      ) -> List[Mapping[str, str]]:
        """
        Convert a prompt to messages object.
        """
        messages = []
        if system_message:
            messages.append(
                {"role": "system", "content": system_message}
            )
        messages.append({"role": "user", "content": prompt})
        
        return messages


class Gemma2FewShot(HFFewShot):
    def __init__(self, 
                model_name: str, 
                model_details: dict=None,
                labels: List[str]=None,
            ):
        
        super().__init__(model_name, model_details, labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.terminators = list(set(t for t in self.terminators if t))
    
    def generate_answer(self, message_object: List[Dict]) -> str:
        """
        Code to process a single list of conversational history and generate the answer.
        """
        # Applying the chat template to the list of messages (conversational history)
        message = self.tokenizer.apply_chat_template(
            message_object,
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
            temperature=self.temperature if self.do_sample else None,
            top_p=0.95 if self.do_sample else None,
            top_k=20 if self.do_sample else None,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decoding the generated text
        answer_text = self.tokenizer.decode(
            outputs[0, model_input['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return answer_text.strip()
        
    def generate_answer_batch(self, message_objects: List[Dict]) -> List[str]:
        
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        """
        messages = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            ) for messages in message_objects
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
            temperature=self.temperature if self.do_sample else None,
            top_p=0.95 if self.do_sample else None,
            top_k=20 if self.do_sample else None,
            eos_token_id=self.terminators,
            output_scores=False
        )

        answer_texts = self.tokenizer.batch_decode(
            outputs[:, model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        answer_texts = [a.strip() for a in answer_texts]
        
        return answer_texts


class LlamaFewShot(HFFewShot):
    def __init__(self, 
                 model_name: str, 
                 model_details: dict=None,
                 labels: List[str]=None
                ):
        
        super().__init__(model_name, model_details, labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.terminators = list(set(t for t in self.terminators if t))
        
    
    def generate_answer(self, message_object: List[Dict]) -> str:
        """
        Code to process a single list of conversational history and generate the answer.
        """
        # Applying the chat template to the list of messages (conversational history)
        message = self.tokenizer.apply_chat_template(
            message_object,
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
            temperature=self.temperature if self.do_sample else None,
            top_p=0.95 if self.do_sample else None,
            top_k=20 if self.do_sample else None,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decoding the generated text
        answer_text = self.tokenizer.decode(
            outputs[0, model_input['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return answer_text.strip()
    
    def generate_answer_batch_logprobs(self, message_objects: List[Dict]) -> Mapping[str, Union[List[str], Tuple[torch.FloatTensor]]]:
    
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        """
        messages = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            ) for messages in message_objects
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
            temperature=self.temperature if self.do_sample else None,
            top_p=0.95 if self.do_sample else None,
            top_k=20 if self.do_sample else None,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True, 
            output_scores=True
        )

        answer_texts = self.tokenizer.batch_decode(
            outputs.sequences[:, model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        return {"answers": answer_texts, "scores": outputs.scores}


    def generate_answer_batch(self, message_objects: List[Dict]) -> List[str]:
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        """
        messages = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            ) for messages in message_objects
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
            temperature=self.temperature if self.do_sample else None,
            top_p=0.95 if self.do_sample else None,
            top_k=20 if self.do_sample else None,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True
        )

        answer_texts = self.tokenizer.batch_decode(
            outputs[:, model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )


        return answer_texts

class Gwen2FewShot(HFFewShot):
    def __init__(self, 
                 model_name: str, 
                 model_details: dict=None,
                 labels: List[str]=None
                ):
        
        super().__init__(model_name, model_details, labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        ]
        self.terminators = list(set(t for t in self.terminators if t))

    def generate_answer(self, message_object: List[Dict]) -> str:
        """
        Code to process a single list of conversational history and generate the answer.
        """
        # Applying the chat template to the list of messages (conversational history)
        message = self.tokenizer.apply_chat_template(
            message_object,
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
            temperature=self.temperature if self.do_sample else None,
            top_p=0.95 if self.do_sample else None,
            top_k=20 if self.do_sample else None,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decoding the generated text
        answer_text = self.tokenizer.decode(
            outputs[0, model_input['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return answer_text.strip()

  
    def generate_answer_batch_logprobs(self, message_objects: List[Dict]) -> Mapping[str, Union[List[str], Tuple[torch.FloatTensor]]]:
        """
        Code to batch process multiple questions.
        """
        messages = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            ) for messages in message_objects
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
            temperature=self.temperature if self.do_sample else None,
            top_p=0.95 if self.do_sample else None,
            top_k=20 if self.do_sample else None,
            eos_token_id=self.terminators,
            return_dict_in_generate=True, 
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True
        )

        answer_texts = self.tokenizer.batch_decode(
            outputs.sequences[:, model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        return {"answers": answer_texts, "scores": outputs.scores}


    def generate_answer_batch(self, message_objects: List[Dict]) -> List[str]:
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        """
        messages = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            ) for messages in message_objects
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
            temperature=self.temperature if self.do_sample else None,
            top_p=0.95 if self.do_sample else None,
            top_k=20 if self.do_sample else None,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True
        )

        answer_texts = self.tokenizer.batch_decode(
            outputs[:, model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        return answer_texts

class Gemma3FewShot(HFFewShot):
    def __init__(self, 
                 model_name: str, 
                 model_details: dict=None,
                 labels: List[str]=None
                ):
        
        super().__init__(model_name, model_details, labels, model_class=Gemma3ForConditionalGeneration)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_turn|>"),
        ]
        self.terminators = list(set(t for t in self.terminators if t))
    
    def _prepare_message_object(self, message_object: List[Dict]) -> List[dict]:
        """
        Convert a message object to the format required by the model.
        """
        # Convert the message object to a list of strings
        message_object = [
            {
                'role': message['role'],
                'content': [{"type": "text", "text": message['content']}] if isinstance(message['content'], str) else message['content']
            } 
            for message in message_object
        ]
        return message_object
    

    def _prepare_message_objects(self, message_objects: List[Dict]) -> List[str]:
        """
        Convert a list of message objects to the format required by the model.
        """
        # Convert the message object to a list of strings
        message_objects = [
            self._prepare_message_object(message_object)
            for message_object in message_objects
        ]
        return message_objects


    def generate_answer(self, message_object: List[Dict]) -> str:
        """
        Code to process a single list of conversational history and generate the answer.
        """
        message_object = self._prepare_message_object(message_object)

        model_input = self.processor.apply_chat_template(
            message_object, add_generation_prompt=True, tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)

        with torch.inference_mode():
            outputs = self.model.generate(
                **model_input,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                eos_token_id=self.terminators,
                temperature=self.temperature if self.do_sample else None,
                top_p=0.95 if self.do_sample else None,
                top_k=20 if self.do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decoding the generated text
        answer_text = self.processor.decode(
            outputs[0][model_input['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return answer_text.strip()
    
    def generate_answer_batch(self, message_objects: List[Dict]) -> List[str]:
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        """
        message_objects = self._prepare_message_objects(message_objects)

        model_inputs = self.processor.apply_chat_template(
            message_objects, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs, 
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                eos_token_id=self.terminators,
                temperature=self.temperature if self.do_sample else None,
                top_p=0.95 if self.do_sample else None,
                top_k=20 if self.do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        answer_texts = self.processor.batch_decode(
            outputs[:, model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        answer_texts = [a.strip() for a in answer_texts]

        return answer_texts
    
    def generate_answer_batch_logprobs(self, message_objects: List[Dict]) -> Mapping[str, Union[List[str], Tuple[torch.FloatTensor]]]:
        """
        Code to batch process multiple questions.
        """
        message_objects = self._prepare_message_objects(message_objects)

        model_inputs = self.processor.apply_chat_template(
            message_objects, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs, 
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_p=0.8 if self.do_sample else None,
                top_k=20 if self.do_sample else None,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True, 
                output_scores=True
            )

        answer_texts = self.processor.batch_decode(
            outputs.sequences[:, model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        answer_texts = [a.strip() for a in answer_texts]

        return {"answers": answer_texts, "scores": outputs.scores}


# class MistralFewShot:
#     def __init__(self, 
#                 model_name: str, 
#                 labels: List[str]=None, 
#                 model_details: dict=None):
        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         # TODO: support quantization
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        
#         self.max_new_tokens = model_details.get("max_new_tokens", 10)
#         self.temperature = model_details.get("temperature", 0.01)
#         self.do_sample = model_details.get("do_sample", True)

#         if labels and model_details["scores"]:
#             self.label_id_map = labels_vocab_id_map(self.tokenizer, labels)
#             print("Label ID Map: ", self.label_id_map)      
#         # TODO: consider setting default for `self.label_id_map` 

#     def debug(self): 
#         while True: 
#             message_object = input("Enter prompt string: ")
#             print(self.generate_answer(message_object))


#     def generate_answer(self, question_text: str):
#         # This is for debug, mostly
        
#         messages = [
#                 {"role": "user", "content": f"{question_text}"},
#             ]

#         encoded_text = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
#         model_inputs = encoded_text.to(self.model.device) # has to be in cuda
#         outputs = self.model.generate(
#             model_inputs, 
#             max_new_tokens=self.max_new_tokens,
#             temperature=self.temperature if self.do_sample else None,
#             # top_p=self.top_p,

#             do_sample=self.do_sample, 
#             pad_token_id=self.tokenizer.eos_token_id,
#         )

#         # skip_special_tokens doesn't work sometimes so we do it manually
#         answer_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#         return answer_text.split("[/INST]")[-1].strip()
    
#     def generate_answer_batch(self, 
#                             message_objects: List[Dict]) -> List[str]:
        
#         """
#         Code to batch process multiple questions. 
#         Can be generalized to other types of query processing.
#         TODO: Get FlashAttention2 to work with this.
#         """
        
#         messages = [
#             self.tokenizer.apply_chat_template(messages, tokenize=False) 
#             for messages in message_objects
#         ]

#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         model_inputs = self.tokenizer(messages, return_tensors="pt", padding=True).to(self.model.device)

#         outputs = self.model.generate(
#             **model_inputs, 
#             max_new_tokens=self.max_new_tokens,
#             do_sample=self.do_sample,
#             temperature=self.temperature, if self.do_sample else None 
#             # top_p=self.top_p,

#             pad_token_id=self.tokenizer.eos_token_id,
#         )

#         answer_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         return [answer_text.split("[/INST]")[-1].strip() for answer_text in answer_texts]
    
#     def generate_answer_batch_scores(self, message_objects: List[Dict]) -> list[str]:

#         if not hasattr(self, "labels"):
#             raise ValueError("Labels not set. Use set_labels() to set labels.")

#         messages = [
#             self.tokenizer.apply_chat_template(messages, tokenize=False) 
#             for messages in message_objects
#         ]
        
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         model_inputs = self.tokenizer(messages, return_tensors="pt", padding=True).to(self.model.device)
        
#         outputs = self.model.generate(
#             **model_inputs, 
#             max_new_tokens=self.max_new_tokens,
#             do_sample=self.do_sample,
#             temperature=self.temperature, if self.do_sample else None 
#             # top_p=self.top_p,

#             pad_token_id=self.tokenizer.eos_token_id,
#             return_dict_in_generate=True,
#             output_scores=True
#         )
        
#         answer_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
#         # do a sanity check on outputs 

#         answer_texts = [answer_text.split("[/INST]")[-1].strip() for answer_text in answer_texts]
#         # TODO: Get the logprobs for the labels
#         # TODO: need `get_label_logprobs` function
#         # stance_logprobs = get_label_logprobs(outputs.scores, self.label_id_map)
#         stance_logprobs = None

#         return answer_texts, stance_logprobs
