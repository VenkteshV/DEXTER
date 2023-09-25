from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, tokenizer: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def tokenize(self, input, **kwargs):
        return self.tokenizer(input, **kwargs)

    def decode(self, idx):
        return self.tokenizer.decode(idx)
