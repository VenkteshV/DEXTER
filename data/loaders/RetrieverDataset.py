import configparser
from typing import Dict, List

from constants import Split
from data.datastructures.evidence import Evidence
from data.datastructures.question import Question
from data.loaders.BaseDataLoader import PassageDataLoader
from data.loaders.DataLoaderFactory import DataLoaderFactory

from data.loaders.Tokenizer import Tokenizer


class RetrieverDataset:
    def __init__(self, dataset:str,passage_dataset:str,config_path,split:Split, batch_size=32,tokenizer="bert-base-uncased"):
        self.split = split
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.tokenizer_name = tokenizer
        if(self.tokenizer_name):
            self.tokenizer = Tokenizer(self.tokenizer_name)
        else:
            self.tokenizer = None
        self.passage_dataloader = PassageDataLoader(passage_dataset,None,self.tokenizer_name,config_path)
        base_dataset = DataLoaderFactory().create_dataloader(dataset, config_path=config_path, split=self.split, batch_size=batch_size,tokenizer=self.tokenizer_name,
                                                             corpus=self.passage_dataloader.raw_data)
        self.base_dataset = base_dataset          
    
    
    def qrels(self)->(List[Question],Dict,List[Evidence]):
        qrels = {}
        queries = []
        corpus = self.passage_dataloader.raw_data
        for sample in self.base_dataset.raw_data:
            if str(sample.question.id()) not in list(qrels.keys()):
                qrels[str(sample.question.id())] = {}
                if(sample.question not in queries):
                    queries.append(sample.question)
            evidence = sample.evidences
            #print("str(sample.idx)",str(sample.idx),str(evidence.id()),qrels[str(sample.idx)])
            qrels[sample.question.id()][str(evidence.id())] = 1
        return queries,qrels,corpus