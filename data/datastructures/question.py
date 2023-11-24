class Question:

    def __init__(self, text:str, idx=None):
        self._text = text
        self._idx = idx
        self.attention_mask = None

    def text(self):
        return self._text

    def id(self):
        return self._idx

    def set_id(self, id):
        self._id = id

    def set_attention_mask(self, attention_mask):
        self.attention_mask = attention_mask
