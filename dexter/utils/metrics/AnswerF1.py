import numpy as np
from dexter.utils.metrics.CoverExactMatch import CoverExactMatch


class AnswerF1:

    def __init__(self):
        self.em = CoverExactMatch()

    def get_f1(self, answers, predictions, is_equal=None):
        '''
        :answers: a list of list of strings
        :predictions: a list of strings
        '''
        assert len(answers)>0 and len(predictions)>0, (answers, predictions)
        occupied_answers = [False for _ in answers]
        occupied_predictions = [False for _ in predictions]
        for i, answer in enumerate(answers):
            for j, prediction in enumerate(predictions):
                if occupied_answers[i] or occupied_predictions[j]:
                    continue
                em = self.em.evaluate(answer, prediction)
                if em:
                    occupied_answers[i] = True
                    occupied_predictions[j] = True
        assert np.sum(occupied_answers)==np.sum(occupied_predictions)
        a, b = np.mean(occupied_answers), np.mean(occupied_predictions)
        if a+b==0:
            return 0
        return 2*a*b/(a+b)