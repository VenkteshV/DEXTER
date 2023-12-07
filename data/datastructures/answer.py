from itertools import chain
from typing import List


class Answer:
    def __init__(self, text: str, idx=None):
        self._text = text
        self._idx = idx

    def text(self):
        return self._text

    def id(self):
        return self._idx

    def flatten(self):
        return [self._text]

    # def set_id(self, id):
    #     self._id = id
    #
    # def set_attention_mask(self, attention_mask):
    #     self.attention_mask = attention_mask


class AmbigNQAnswer:
    def __init__(self, answers: List[List[List[Answer]]]):
        self.answers = answers

    def flatten(self):
        flattened_answers = []
        for annotation in self.answers:
            for query in annotation:
                for answer in query:
                    flattened_answers.append(answer.text())
        return flattened_answers

class TATQAAnswer(Answer):

    def __init__(self, answers: List[Answer], idx=None):
        self.answers = answers

    def text(self):
        return ",".join(self._text)

    def id(self):
        return self._idx
