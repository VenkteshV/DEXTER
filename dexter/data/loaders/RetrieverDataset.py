import configparser
from typing import Dict, List

from dexter.config.constants import Split
from dexter.data.datastructures.evidence import Evidence
from dexter.data.datastructures.question import Question
from dexter.data.loaders.BaseDataLoader import PassageDataLoader
from dexter.data.loaders.DataLoaderFactory import DataLoaderFactory

from dexter.data.loaders.Tokenizer import Tokenizer


class RetrieverDataset:
    '''Dataset class to load the data of the corresponding dataset provided for the evaluation of retrieval.
    Arguments
    dataset (str): alias of dataset
    passage_dataset (str): alias of corpus
    config_path (str) : path to the configuration file containing various parameters
    split (Split) : Split of the dataset to be loaded
    batch_size (int) : batch size to process the dataset.
     tokenzier (str) : name of the tokenizer model. Set tokenizer as None, if only samples to be loaded but not tokenized and stored. This can help save time if only the raw dataset is needed.
    '''
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