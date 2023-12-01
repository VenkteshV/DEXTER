import json
from constants import Split
from data.loaders.DataLoaderFactory import DataLoaderFactory
from data.loaders.FinQADataLoader import FinQADataLoader

loader_factory = DataLoaderFactory()

loader_dev = loader_factory.create_dataloader("tatqa",config_path="tests/data/test_config.ini", split=Split.DEV, batch_size=10)
loader_train = loader_factory.create_dataloader("tatqa",config_path="tests/data/test_config.ini", split=Split.TRAIN, batch_size=10)
#loader_test = loader_factory.create_dataloader("tatqa",config_path="tests/data/test_config.ini", split=Split.TEST, batch_size=10)
corpus = {}

for data in loader_dev.raw_data+loader_train.raw_data:
    if(data.evidences.id() not in corpus.keys()):
        ev = {"passage":data.evidences.text(),"title":data.evidences.title()}
        corpus[data.evidences.id()] = ev


with open("tests/corpus_utils/tatqa_corpus.json","w+") as fp:
    json.dump(corpus,fp)
