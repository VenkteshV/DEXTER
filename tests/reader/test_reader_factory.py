import unittest

from readers.ReaderFactory import ReaderFactory, ReaderName


class MyTestCase(unittest.TestCase):
    def test_bert_reader(self):
        reader = ReaderFactory().create_reader(reader_name=ReaderName.BERT.name,
                                               base="bert-base-uncased",
                                               checkpoint="tests/data/reader/data/best-model.pt"
                                               )


if __name__ == '__main__':
    unittest.main()
