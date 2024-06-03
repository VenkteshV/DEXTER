import json
from config.constants import DataTypes, Split
from data.datastructures.evidence import TableEvidence
from data.loaders.FinQADataLoader import FinQADataLoader


loader_dev = FinQADataLoader("finqa", config_path="tests/corpus_utils/config.ini", split=Split.DEV, batch_size=10)
loader_train = FinQADataLoader("finqa", config_path="tests/corpus_utils/config.ini", split=Split.TRAIN, batch_size=10)
loader_test = FinQADataLoader("finqa", config_path="tests/corpus_utils/config.ini", split=Split.TEST, batch_size=10)
corpus = {}

for data in loader_dev.raw_data+loader_train.raw_data+loader_test.raw_data:
    ev = {"passage":data.evidences.text(),"title":data.evidences.title()}
    if(isinstance(data.evidences,TableEvidence)):
        ev["type"] = DataTypes.TABLE
        
    if(data.evidences.id() in corpus.keys()):
        print(data.evidences.id()+"--err")
    corpus[data.evidences.id()] = ev


with open("tests/corpus_utils/dataset/finqa/finqa_corpus.json","w+") as fp:
    json.dump(corpus,fp)


