import unittest

from constants import Split
from data.loaders.RetrieverDataset import RetrieverDataset
from metrics.SimilarityMatch import CosineSimilarity
from retriever.ANCE import ANCE

import pytrec_eval
import json




class MyTestCase(unittest.TestCase):
    def test_retriever_dataloader(self):
        loader = RetrieverDataset("finqa","finqa-corpus","tests/data/test_config.ini",Split.DEV)
        retriever = ANCE("tests/retriever/test_config.ini")
        queries, qrels, corpus = loader.qrels()
        qrels_ret = retriever.retrieve(corpus,queries,10,CosineSimilarity(),True)
        self.assertEqual(len(qrels),len(qrels_ret))
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg'})
        results = evaluator.evaluate(qrels_ret)
        # Calculate overall NDCG@10
        total_ndcg_at_10 = 0
        num_queries = len(qrels_ret)

        # Sum up NDCG@10 values for all queries
        for query_id, query_results in results.items():
            total_ndcg_at_10 += query_results['ndcg']

        # Calculate average NDCG@10
        average_ndcg_at_10 = total_ndcg_at_10 / num_queries

        print(f"Overall NDCG@10: {average_ndcg_at_10}")



if __name__ == '__main__':
    unittest.main()