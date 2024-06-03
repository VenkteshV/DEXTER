import json
import os
from dexter.config.constants import Split
from dexter.data.datastructures.dataset import DprDataset
from dexter.data.datastructures.evidence import Evidence, TableEvidence
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.sample import Sample
from dexter.data.loaders.BaseDataLoader import GenericDataLoader
from dexter.config.constants import DataTypes
from dexter.config.constants import DataTypes


class TATQADataLoader(GenericDataLoader):
    '''Data loader class to load Datset from raw TATQA dataset.
    TATQA dataset consists of a questions each of which have a single answer and variopus table and text evidences.
    
    Arguments:
    dataset (str): string containing the dataset alias
    tokenzier (str) : name of the tokenizer model. Set tokenizer as None, if only samples to be loaded but not tokenized and stored. This can help save time if only the raw dataset is needed.
    config_path (str) : path to the configuration file containing various parameters
    split (Split) : Split of the dataset to be loaded
    batch_size (int) : batch size to process the dataset.
    corpus Dict[str,Evidence]: corpus containing all needed passages.    
    '''
    
    def __init__(self, dataset: str, tokenizer="bert-base-uncased", config_path='test_config.ini', split=Split.TRAIN,
                 batch_size=None, corpus=None):
        super().__init__(dataset, tokenizer, config_path, split, batch_size)
        self.corpus = corpus

    def load_raw_dataset(self, split=Split.TRAIN):
        dataset = self.load_json(split)
        for data in dataset:
            for question in data["questions"]:
                q = Question(question["question"],question["uid"])
                count = 0
                
                if(DataTypes.TABLE in question["answer_from"]):
                    rows = [list(row) for row in zip(*data["table"]["table"])]
                    columns = rows[0]
                    table = rows[1:]
                    self.raw_data.append(Sample(q.id()+"-"+str(count), q, question["answer"], TableEvidence(table=table,columns=columns,idx=data["table"]["uid"])))
                    count+=1
                if(DataTypes.TEXT in question["answer_from"]):
                    for paragraph_id in question["rel_paragraphs"]:
                        self.raw_data.append(Sample(q.id()+"-"+str(count), q, question["answer"], Evidence(data["paragraphs"][int(paragraph_id)-1]["text"],data["paragraphs"][int(paragraph_id)-1]["uid"])))
                        count+=1

    def load_tokenized(self):
        if self.tokenized_path and os.path.exists(self.tokenized_path):
            self.logger.info("Loading TATQA data from {}".format(self.tokenized_path))
            with open(self.tokenized_path, "r") as f:
                ip_ids, ip_attention, evidence_ids, evidence_attention = json.load(f)
        else:
            ip_ids, ip_attention = self.tokenize_questions()
            evidence_ids, evidence_attention = self.tokenize_evidences()
        return DprDataset(ip_ids, ip_attention, evidence_ids, evidence_attention)