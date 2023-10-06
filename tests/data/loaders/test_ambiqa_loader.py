import unittest

from constants import Split
from data.datastructures.dataset import QaDataset
from data.datastructures.sample import AmbigNQSample
from data.loaders.BasedataLoader import AmbigQADataLoader
from data.loaders.tokenizer import Tokenizer


class MyTestCase(unittest.TestCase):
    def test_loader(self):
        loader = AmbigQADataLoader("ambignq-light", config_path="test_config.ini", split=Split.TRAIN, batch_size=10)
        assert len(loader.raw_data) == len(loader.dataset)
        self.assertTrue(isinstance(loader.dataset, QaDataset))
        self.assertTrue(isinstance(loader.tokenizer, Tokenizer))
        self.assertTrue(isinstance(loader.raw_data[0], AmbigNQSample))



if __name__ == '__main__':
    unittest.main()
