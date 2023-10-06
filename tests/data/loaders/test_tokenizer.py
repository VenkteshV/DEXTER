import unittest

from data.loaders.tokenizer import Tokenizer


class MyTestCase(unittest.TestCase):
    def test_tokenizer(self):
        tokenizer = Tokenizer("bert-base-uncased")
        op = tokenizer.tokenize(["Hello"])['input_ids']
        self.assertEqual(len(op), 1)
        self.assertEqual("[CLS] hello [SEP]", tokenizer.decode(op[0]))


if __name__ == '__main__':
    unittest.main()
