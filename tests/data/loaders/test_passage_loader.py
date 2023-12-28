import unittest

from data.loaders.BaseDataLoader import PassageDataLoader


class MyTestCase(unittest.TestCase):
    def test_loader(self):
        loader = PassageDataLoader(dataset="ottqa-corpus",subset_ids=None,config_path="tests/data/test_config.ini",tokenizer=None)
        self.assertIsNotNone(loader)
        #self.assertEqual(len(loader.dataset),2)



if __name__ == '__main__':
    unittest.main()
