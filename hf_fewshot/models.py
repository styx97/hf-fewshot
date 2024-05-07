import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

def labels_vocab_id_map(tokenizer, labels): 
    label_id_map = {}
    for label in labels: 
        ids = tokenizer.encode(label, skip_special_tokens=True)
        if len(ids) > 1: 
            raise ValueError("Label should not be breakable")
            # TODO: For breakable words, logprob is just the sum of subword logprobs
            # implement that sometime later
        label_id_map[label] = ids[1]
    
    return label_id_map

def get_logsoftmax(x): 
    m = torch.nn.LogSoftmax(dim=1)
    return m(x.view(1, -1))

def get_label_logprobs(scores, label_id_map):
    batch_size = scores[0].shape[0]
    batch_logprobs = [] 
    for index in range(batch_size): 
        # look at the first output token of the generation 
        # and get the logprob of the label
        relevant_position = scores[0]
        token_output_logprobs = get_logsoftmax(relevant_position[index])
        numpy_formatted = token_output_logprobs[0].detach().cpu().numpy()
        label_logprobs=  {
            k:float(numpy_formatted[v]) for k, v in label_id_map.items()
        }

        batch_logprobs.append(label_logprobs)

    assert len(batch_logprobs) == batch_size, "Batch size mismatch"
    return batch_logprobs

class GPTTurbo:
    """
    a class the calls an openai model to generate text
    The openai key is fetched from the env variable OPENAI_API_KEY
    """


class HFFewShot:
    def __init__(self,
                model_name: str, 
                model_details: dict=None):
        
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
        
        self.max_new_tokens = model_details.get("max_new_tokens", 10)
        self.temperature = model_details.get("temperature", 0.01)


    def debug(self): 
        while True: 
            query_text = input("Enter prompt string: ")
            print(self.generate_answer(query_text))

    def generate_answer_batch(self, 
                        query_texts: list) -> list[str]: 
    
        """
        Code to batch process multiple questions. 
        Can be generalized to other types of query processing.
        TODO: Get FlashAttention2 to work with this.
        """
        
        messages = [
            self.tokenizer.apply_chat_template(messages,
                                                tokenize=False) 
                                                for messages in query_texts]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        model_inputs = self.tokenizer(messages,
                                    return_tensors="pt",
                                    padding=True).to("cuda")


        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens = self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature, 
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answer_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return [answer_text.split("[/INST]")[-1].strip() for answer_text in answer_texts]
        
class LlamaFewShot(HFFewShot):
    def __init__(self, 
                model_name: str, 
                model_details: dict=None):
        
        super().__init__(model_name, model_details)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate_answer_batch(self, 
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
        ).to("cuda")

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens = self.max_new_tokens,
            do_sample=True,
            eos_token_id=terminators,
            temperature=self.temperature, 
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answer_texts = self.tokenizer.batch_decode(outputs[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return answer_texts

class MistralFewShot:
    def __init__(self, 
                model_name: str, 
                labels: list[str]=None, 
                model_details: dict=None):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,                                                    
                                                         torch_dtype=torch.bfloat16,
                                                         device_map="auto")
        
        self.max_new_tokens = model_details.get("max_new_tokens", 10)
        self.temperature = model_details.get("temperature", 0.01)

        if labels and model_details["scores"]:
            self.label_id_map = labels_vocab_id_map(self.tokenizer, labels)
            print("Label ID Map: ", self.label_id_map)
            

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
        model_inputs = encoded_text.to("cuda") # has to be in cuda
        outputs = self.model.generate(
            model_inputs, 
            max_new_tokens = 10,
            temperature=0.01,
            do_sample=True, 
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
            self.tokenizer.apply_chat_template(messages,
                                                tokenize=False) 
                                                for messages in query_texts]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        model_inputs = self.tokenizer(messages,
                                    return_tensors="pt",
                                    padding=True).to("cuda")

        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens = self.max_new_tokens,
            do_sample=True,
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
            self.tokenizer.apply_chat_template(messages,
                                                tokenize=False) 
                                                for messages in query_texts
        ]
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model_inputs = self.tokenizer(messages,
                                    return_tensors="pt",
                                    padding=True).to("cuda")
        
        outputs = self.model.generate(**model_inputs, 
                                    max_new_tokens=self.max_new_tokens,
                                    do_sample=True,
                                    temperature=self.temperature, 
                                    pad_token_id=self.tokenizer.eos_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True)
        
        answer_texts = self.tokenizer.batch_decode(outputs.sequences,
                                                    skip_special_tokens=True)
        
        # do a sanity check on outputs 

        answer_texts = [answer_text.split("[/INST]")[-1].strip() for answer_text in answer_texts]
        stance_logprobs = get_label_logprobs(outputs.scores, self.label_id_map)

        return answer_texts, stance_logprobs