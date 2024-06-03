import unittest

from dexter.config.constants import Split
from dexter.data.datastructures.dataset import QADataset
from dexter.data.datastructures.sample import AmbigNQSample
from dexter.data.loaders.AmbigQADataLoader import AmbigQADataLoader
from dexter.data.loaders.Tokenizer import Tokenizer


class MyTestCase(unittest.TestCase):
    def test_loader(self):
        loader = AmbigQADataLoader("ambignq-light", config_path="tests/data/test_config.ini", split=Split.DEV, batch_size=10)
        assert len(loader.raw_data) == len(loader.dataset)
        self.assertTrue(isinstance(loader.dataset, QADataset))
        self.assertTrue(isinstance(loader.tokenizer, Tokenizer))
        self.assertTrue(isinstance(loader.raw_data[0], AmbigNQSample))



if __name__ == '__main__':
    unittest.main()
