import unittest

from constants import Split
from data.loaders.RetrieverDataset import RetrieverDataset




class MyTestCase(unittest.TestCase):
    def test_retriever_dataloader(self):
        loader = RetrieverDataset("finqa","finqa-corpus","tests/data/test_config.ini",Split.DEV)
        self.assertIsNotNone(loader.qrels)
        self.assertEqual(len(loader.qrels.keys()),3)



if __name__ == '__main__':
    unittest.main()