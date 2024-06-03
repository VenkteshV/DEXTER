import unittest

from dexter.config.constants import Split
from dexter.data.datastructures.dataset import DprDataset
from dexter.data.datastructures.evidence import Evidence
from dexter.data.datastructures.sample import Sample
from dexter.data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader
from dexter.data.loaders.Tokenizer import Tokenizer


class MyTestCase(unittest.TestCase):
    def test_loader(self):
        loader = WikiMultihopQADataLoader("wikimultihopqa", config_path="tests/data/test_config.ini", split=Split.DEV, batch_size=10)
        assert len(loader.raw_data) == len(loader.dataset)
        self.assertTrue(isinstance(loader.dataset, DprDataset))
        self.assertTrue(isinstance(loader.tokenizer, Tokenizer))
        self.assertTrue(isinstance(loader.raw_data[0], Sample))
        self.assertTrue(isinstance(loader.raw_data[0].evidences, Evidence))



if __name__ == '__main__':
    unittest.main()
