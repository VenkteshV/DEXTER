import json
import os
from dexter.config.constants import Split
from dexter.data.datastructures.answer import Answer
from dexter.data.datastructures.dataset import DprDataset
from dexter.data.datastructures.evidence import Evidence, TableEvidence
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.sample import Sample
from dexter.data.loaders.BaseDataLoader import GenericDataLoader


class FinQADataLoader(GenericDataLoader):
    '''Data loader class to load Datset from raw FinQA dataset.
    FinQA dataset consists of a questions each of which have a single answer and three contexts - pre text, post text and table.
    
    Arguments:
    dataset (str): string containing the dataset alias
    tokenzier (str) : name of the tokenizer model. Set tokenizer as None, if only samples to be loaded but not tokenized and stored. This can help save time if only the raw dataset is needed.
    config_path (str) : path to the configuration file containing various parameters
    split (Split) : Split of the dataset to be loaded
    batch_size (int) : batch size to process the dataset.
    corpus: corpus containing all needed passages.    
    '''
        
    def __init__(self, dataset: str, tokenizer="bert-base-uncased", config_path='test_config.ini', split=Split.TRAIN,
                 batch_size=None, corpus=None):
        super().__init__(dataset, tokenizer, config_path, split, batch_size)

    def load_raw_dataset(self, split=Split.TRAIN):
        print("split",split)
        dataset = self.load_json(split)
        for data in dataset:
            question = Question(data["qa"]["question"],idx=data["id"])
            answer = Answer(data["qa"]["answer"],idx=data["id"])
            self.raw_data.append(Sample(data["id"]+"-0", question, answer, TableEvidence(data["table"][1:],columns=data["table"][0],idx=data["id"]+".table")))
            self.raw_data.append(Sample(data["id"]+"-1", question, answer, Evidence("".join(data["pre_text"]),idx=data["id"]+".pre")))
            self.raw_data.append(Sample(data["id"]+"-2", question, answer, Evidence("".join(data["post_text"]),idx=data["id"]+".post")))

    def load_tokenized(self):
        if self.tokenized_path and os.path.exists(self.tokenized_path):
            self.logger.info("Loading DPR data from {}".format(self.tokenized_path))
            with open(self.tokenized_path, "r") as f:
                ip_ids, ip_attention, evidence_ids, evidence_attention = json.load(f)
        else:
            ip_ids, ip_attention = self.tokenize_questions()
            evidence_ids, evidence_attention = self.tokenize_evidences()
        return DprDataset(ip_ids, ip_attention, evidence_ids, evidence_attention)