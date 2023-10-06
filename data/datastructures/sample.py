from typing import List

from typing import Optional

from data.datastructures.answer import Answer, AmbigNQAnswer
from data.datastructures.question import Question


class Sample:
    def __init__(self, _id, question: Question, answer: Answer, evidences: Optional = None):
        self.question = question
        self.evidences = evidences
        self.answer = answer


class AmbigNQSample:
    def __init__(self, _id, question: Question, answers: AmbigNQAnswer, evidences: Optional = None):
        self.question = question
        self.evidences = evidences
        self.answer = answers
