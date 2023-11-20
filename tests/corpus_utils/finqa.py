import json
from constants import Split
from data.loaders.FinQADataLoader import FinQADataLoader


loader = FinQADataLoader("finqa", config_path="tests/data/test_config.ini", split=Split.DEV, batch_size=10)
corpus = {}
for data in loader.raw_data:
    ev = {"passage":data.evidences.text(),"title":data.evidences.title()}
    corpus[data.evidences.id()] = ev

with open("tests/corpus_utils/finqa_corpus.json","w+") as fp:
    json.dump(corpus,fp)


