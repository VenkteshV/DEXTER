import unittest

from constants import Split
from data.datastructures.dataset import QADataset
from data.datastructures.sample import AmbigNQSample
from data.loaders import ReaderDataLoader
from data.loaders.BaseDataLoader import AmbigQADataLoader, PassageDataLoader
from data.loaders.Tokenizer import Tokenizer
from readers.ReaderFactory import ReaderFactory, ReaderName
from metrics.ExactMatch import ExactMatch


class MyTestCase(unittest.TestCase):
    def test_ambignqreader(self):
        loader = ReaderDataLoader("ambignq-light","wiki-100","tests/data/test_config.ini",Split.DEV)
        reader = ReaderFactory().create_reader(reader_name=ReaderName.BERT.name,
                                               base="bert-base-uncased",
                                               checkpoint="tests/data/reader/data/best-model.pt"
                                               )
        outputs = reader.evaluate(loader,[ExactMatch()])
        self.assertEqual(outputs,{'Exact Match': 0.6363636363636364})



if __name__ == '__main__':
    unittest.main()
