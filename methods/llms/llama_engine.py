import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaEngine:
    def __init__(self, data, model_name="meta-llama/Llama-2-7b-hf", temperature=None, top_n=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.cuda()
        self.temperature = temperature
        self.data = data
        self.top_n = top_n

    def get_llama_completion(self, prompt: str):

        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_length=1000,
            do_sample=True,
            top_k=self.top_n,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        output = pipeline(prompt)
        return output[0]["generated_text"]
