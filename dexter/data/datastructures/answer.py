from typing import List


class Answer:
    """
    A base class to hold all details about the answer aspect in question answering.

    Attributes:
        _text (str): The text of the answer.
        _idx (int): The ID of the answer.
    """
    def __init__(self, text: str, idx=None):
        self._text = text
        self._idx = idx

    def text(self)->str:
        return self._text

    def id(self):
        return self._idx

    def flatten(self)->List[str]:
        """
        Flattends the answer structure if complex into a simple list of answer texts.
        """
        return [self._text]

    # def set_id(self, id):
    #     self._id = id
    #
    # def set_attention_mask(self, attention_mask):
    #     self.attention_mask = attention_mask


class AmbigNQAnswer:
    """
    The Ambiguous question answer case contains questions which can be divided into sub-clarified questions with corresponding answers.
    Additionally, each question can have multiple annotations. 
    The element answers[i][j][k] represents the answer given by the ith annotator for the jth clarified question and the kth answer.

    Attributes:
        answers  List[List[List[Answer]]]: The text of the answer given in structure mentioned above.
        _idx id of the answer
    """
    def __init__(self, answers: List[List[List[Answer]]],idx=None):
        self.answers = answers
        self._idx = idx
    
    def id(self):
        return self._idx

    def flatten(self)->List[str]:
        """
        Flattends the answer structure into a simple list of answer texts.
        """
        flattened_answers = []
        for annotation in self.answers:
            for query in annotation:
                for answer in query:
                    flattened_answers.append(answer.text())
        return flattened_answers

class TATQAAnswer(Answer):
    """
    The TATQAAnswer question answer case contains questions qhich contains a sequnce of answers for one question.
    The element answers[i] represents the ith answer for the question.

    Attributes:
        answers  List[Answer]: The text of the answer given in structure mentioned above.
        _idx id of the answer
    """

    def __init__(self, answers: List[Answer], idx=None):
        self.answers = answers
        self.idx = idx

    def text(self):
        return ",".join(self._text)

    def id(self):
        return self._idx
