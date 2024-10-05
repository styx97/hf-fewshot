from simple_colors import *
from hf_fewshot.models import FewShotModel, GPTFewShot, LlamaFewShot
import argparse

class ChatAgent:
    def __init__(self, model_instance: FewShotModel):
        self.model_instance = model_instance
        self.history = []

    def add_to_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def user_input_terminal(self) -> str:
        input_text = input(red("You: "))
        return input_text
    
    def get_response(self, query_text: str) -> str:
        self.add_to_history("user", query_text)
        
        response_content = self.model_instance.generate_answer(self.history)
        self.add_to_history("assistant", response_content)

        return response_content

    def chat_terminal(self):
        print(red("To end the conversation, type 'exit'."))
        while True:
            user_input = self.user_input_terminal()
            if user_input.lower() == 'exit':
                break
            response = self.get_response(user_input)
            print(blue(f"Assistant: {response}"))

    def chat_programmatic(self, input_function: callable):
        """
        Chat programmatically by passing an input function that
        generates user inputs one at a time and handles the response.
        """
        while True:
            user_input = input_function()
            if user_input.lower() == 'exit':
                break
            response = self.get_response(user_input)
            print(blue(f"Assistant: {response}"))

def add_arguments(): 
    parser = argparse.ArgumentParser(description="Chat with a FewShot model")
    
    parser.add_argument(
        '--name', type=str, default='gpt-3.5-turbo', help="Set the model name"
    )

    parser.add_argument(
        '--temperature', type=float, default=0.7, help="Set the temperature for the model"
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=150, help="Set the maximum number of new tokens"
    )
    parser.add_argument(
        '--top_p', type=float, default=1, help="Set the top-p value"
    )

    parser.add_argument(
        '--quantization', type=str, default="4bit", help="Set quantization"
    )
    
    return parser


def main():
    parser = add_arguments()

    args = parser.parse_args()
    model_details = {
        "model_name": args.name,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p, 
        "quantization": args.quantization,
    }

    if 'gpt' in args.name:
        gpt_few_shot = GPTFewShot(model_name=args.name, model_details=model_details)
        model_instance = gpt_few_shot

    elif 'llama' in args.name:
        llama_few_shot = LlamaFewShot(model_name=args.name, model_details=model_details)

        model_instance = llama_few_shot


    # Use the desired model instance for ChatAgent
    model_instance = llama_few_shot
  
    chat_agent = ChatAgent(model_instance)
    chat_agent.chat_terminal()