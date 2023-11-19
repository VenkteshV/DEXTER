import unittest

from constants import Split
from data.loaders.RetrieverDataset import RetrieverDataset
from metrics.SimilarityMatch import CosineSimilarity
from retriever.ANCE import ANCE




class MyTestCase(unittest.TestCase):
    def test_retriever_dataloader(self):
        loader = RetrieverDataset("finqa","finqa-corpus","tests/data/test_config.ini",Split.DEV)
        retriever = ANCE("tests/retriever/test_config.ini")
        queries, qrels, corpus = loader.qrels()
        qrels_ret = retriever.retrieve(corpus,queries,1,CosineSimilarity(),True)
        self.assertEqual(len(qrels),len(qrels_ret))



if __name__ == '__main__':
    unittest.main()