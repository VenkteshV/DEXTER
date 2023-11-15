from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, tokenizer: str,prefix:str=None):
        self.prefix = prefix if prefix else ""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def tokenize(self, input, **kwargs):
        return self.tokenizer(input, **kwargs)

    def decode(self, idx):
        return self.tokenizer.decode(idx,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip().replace(" - ", "-").replace(" : ", ":")
