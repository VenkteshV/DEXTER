from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class FlanT5Engine:
    def __init__(self, data,model_name="google/flan-t5-xl", temperature=None, top_n=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.cuda()
        self.temperature = temperature
        self.data = data

    def get_flant5_completion(self, prompt: str):
        input = self.tokenizer(prompt, return_tensors="pt",
                               max_length=1096).to("cuda")
        output = self.model.generate(**input, max_length=1000)
        decoded_outputs = self.tokenizer.batch_decode(output,
                                                      skip_special_tokens=True, max_length=1000)[0]
        return decoded_outputs
