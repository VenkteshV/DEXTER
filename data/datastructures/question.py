class Question:
    """
    A base class to hold all details about the question aspect in question answering.

    Attributes:
        _text (str): The text of the question.
        _idx (int): The ID of the question.
    """

    def __init__(self, text:str, idx=None):
        self._text = text
        self._idx = idx
        self.attention_mask = None

    def text(self):
        return self._text

    def id(self):
        return self._idx

    def set_id(self, idx):
        self._id = idx

    def set_attention_mask(self, attention_mask):
        self.attention_mask = attention_mask
