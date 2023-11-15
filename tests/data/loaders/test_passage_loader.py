import unittest

from data.loaders.BasedataLoader import PassageDataLoader


class MyTestCase(unittest.TestCase):
    def test_loader(self):
        loader = PassageDataLoader(dataset="wiki-100",subset_ids=[14513,23160],config_path="tests/data/test_config.ini")
        self.assertIsNotNone(loader)
        self.assertEqual(len(loader.dataset),2)



if __name__ == '__main__':
    unittest.main()
