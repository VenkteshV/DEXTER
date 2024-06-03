import numpy as np
import re
import string
from dexter.utils.metrics.MetricsBase import Metric
'''This file contains function to calculate eact match between answers '''


class CoverExactMatch(Metric):
    def normalize_answer(self,s):

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    def name(self):
        return "Exact Match"    
    def evaluate(self,answers1, answers2):
        if type(answers1)==list:
            if len(answers1)==0:
                return 0
            return np.max([self.evaluate(a, answers2) for a in answers1])
        if type(answers2)==list:
            if len(answers2)==0:
                return 0
            return np.max([self.evaluate(answers1, a) for a in answers2])
        return (self.normalize_answer(answers1) in self.normalize_answer(answers2))
