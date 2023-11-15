import unittest

from data.loaders.Tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        tokenizer = Tokenizer("bert-base-uncased")
        op = tokenizer.tokenize(["Hello"])['input_ids']
        self.assertEqual(len(op), 1)
        self.assertEqual("hello", tokenizer.decode(op[0]))


if __name__ == '__main__':
    unittest.main()
