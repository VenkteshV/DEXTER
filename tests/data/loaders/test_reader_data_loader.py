import unittest

from constants import Split
from data.datastructures.dataset import QADataset
from data.datastructures.sample import AmbigNQSample
from data.loaders.BasedataLoader import AmbigQADataLoader, PassageDataLoader, ReaderDataLoader, SeqGenDataLoader
from data.loaders.Tokenizer import Tokenizer


class MyTestCase(unittest.TestCase):
    def test_loader(self):
        loader = SeqGenDataLoader("ambignq-light","wiki-100","tests/data/test_config.ini",Split.DEV)
        self.assertEquals(len(loader.dataset),11)




if __name__ == '__main__':
    unittest.main()