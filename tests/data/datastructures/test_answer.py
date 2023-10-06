import unittest

from data.datastructures.answer import Answer, AmbigNQAnswer


class MyTestCase(unittest.TestCase):
    def test_ambigqa_flatten(self):
        answer = AmbigNQAnswer([[[Answer("Why?"), Answer("Where?")], [Answer("Who?")]], [[Answer("When?")]]])
        flattened = answer.flatten()
        assert len(flattened) == 4
        self.assertTrue(isinstance(flattened[0],str))


if __name__ == '__main__':
    unittest.main()
