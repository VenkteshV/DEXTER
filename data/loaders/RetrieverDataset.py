import configparser

from constants import Split
from data.loaders.BasedataLoader import PassageDataLoader
from data.loaders.DataLoaderFactory import DataLoaderFactory

from data.loaders.Tokenizer import Tokenizer


class RetrieverDataset:
    def __init__(self, dataset:str,passage_dataset:str,config_path,split:Split, batch_size=32,tokenizer="bert-base-uncased"):
        self.split = split
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.tokenizer_name = tokenizer
        self.tokenizer = Tokenizer(self.tokenizer_name)
        base_dataset = DataLoaderFactory().create_dataloader(dataset, config_path, self.split, batch_size,self.tokenizer_name)
        self.base_dataset = base_dataset          
        self.passage_dataloader = PassageDataLoader(passage_dataset,None,self.tokenizer_name,config_path)
        self.qrels = self.load_qrels()
    
    
    def load_qrels(self):
        qrels = {}
        for sample in self.base_dataset.raw_data:
            if str(sample.idx) not in list(qrels.keys()):
                qrels[str(sample.idx)] = {}
            evidence = sample.evidences
            #print("str(sample.idx)",str(sample.idx),str(evidence.id()),qrels[str(sample.idx)])
            qrels[str(sample.idx)][str(evidence.id())] = 1
        return qrels