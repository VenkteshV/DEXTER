'''This file contains function to calculate Bert Score between answers '''
import numpy as np
import re
import string
from dexter.utils.metrics.MetricsBase import Metric
from datasets import load_metric
from statistics import mean,median
import pandas as pd


bertscore_metric = load_metric('bertscore')


class BertScore(Metric):

    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name

    def get_bert_scores(self, answer1, answer2):
        """Compute bertscore between reference and generated answers

        Args:
            answer1 (str): generated answers
            answer2 (str): reference answers

        Returns:
            dict: scores
        """        
        bert_scores = bertscore_metric.compute(predictions=[answer1], references=[answer2], lang="en",model_type=self.model_name)
        print("f1",mean(bert_scores['f1']))
        print("recall:",mean(bert_scores['recall']))
        print("precision:",mean(bert_scores['precision']))
        final_data = {"answer1":answer1,
        "answer2":answer2,
        "P":bert_scores['precision'],
        "R":bert_scores['recall'],
        "F1":bert_scores['f1']}
        return final_data   

    def evaluate(self,answer1, answer2):
        return self.get_bert_scores(answer1, answer2)
