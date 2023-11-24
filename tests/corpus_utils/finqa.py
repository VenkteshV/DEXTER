import json
from constants import Split
from data.loaders.FinQADataLoader import FinQADataLoader


loader_dev = FinQADataLoader("finqa", config_path="tests/data/test_config.ini", split=Split.DEV, batch_size=10)
loader_train = FinQADataLoader("finqa", config_path="tests/data/test_config.ini", split=Split.TRAIN, batch_size=10)
loader_test = FinQADataLoader("finqa", config_path="tests/data/test_config.ini", split=Split.TEST, batch_size=10)
corpus = {}

for data in loader_dev.raw_data+loader_train.raw_data+loader_test.raw_data:
    ev = {"passage":data.evidences.text(),"title":data.evidences.title()}
    if(data.evidences.id() in corpus.keys()):
        print(data.evidences.id()+"--err")
    corpus[data.evidences.id()] = ev


with open("tests/corpus_utils/finqa_corpus.json","w+") as fp:
    json.dump(corpus,fp)


