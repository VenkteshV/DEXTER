from typing import Optional

from dexter.data.datastructures.answer import Answer, AmbigNQAnswer
from dexter.data.datastructures.evidence import Evidence
from dexter.data.datastructures.question import Question


class Sample:
    """
    A base class to hold one datapoint/sample with a question ans its corresponding answer

    Attributes:
        question (Question): question of the sample.
        answer (Answer): answer of the given question.
        evidenc (Answer): Optional context/evidence for the given question.
        _idx (int): The ID of the answer.
    """
    def __init__(self, idx, question: Question, answer: Answer, evidences: Optional[Evidence] = None):
        self.question = question
        self.evidences = evidences
        self.answer = answer
        self.idx = idx


class AmbigNQSample:
    """
    A base class to hold one datapoint/sample with a question ans its corresponding answer for the ambiguous case.

    Attributes:
        question (Question): question of the sample.
        answer (AmbigNQAnswer): answer of the given question.
        evidence (Answer): Optional context/evidence for the given question.
        _idx (int): The ID of the answer.
    """
    def __init__(self, idx, question: Question, answers: AmbigNQAnswer, evidences: Optional[Evidence] = None):
        self.question = question
        self.evidences = evidences
        self.answer = answers
        self.idx = idx
