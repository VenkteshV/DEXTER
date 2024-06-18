import torch
import transformers
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
access_token_read = os.environ["huggingface_token"]
login(token = access_token_read)

class LlamaEngine:

    def __init__(self, data, model_name="meta-llama/Llama-2-7b-chat-hf", temperature=0.3, top_n=1, max_new_tokens=256):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = temperature
        self.data = data
        self.top_n = top_n
        self.max_new_tokens=max_new_tokens
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def get_llama_completion(self, system_prompt: str, user_prompt: str):
        messages = [
            {
                "role": "user",
                "content": system_prompt,
            },
              {
                "role": "assistant",
                "content": "Yes I will reason and generate the answer",
            },
            {"role": "user", "content": user_prompt
},
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            num_return_sequences=1,
            temperature=self.temperature,
            top_k=10,
            top_p=0.95)
        return outputs[0]["generated_text"]