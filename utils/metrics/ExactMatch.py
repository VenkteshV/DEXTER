import numpy as np

from utils.metrics.MetricsBase import Metric
'''This file contains function to calculate eact match between answers '''

class ExactMatch(Metric):
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
        return (self.normalize_answer(answers1) == self.normalize_answer(answers2))
