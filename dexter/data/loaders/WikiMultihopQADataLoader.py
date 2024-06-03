import json
import os
import tqdm
from dexter.config.constants import Split
from dexter.data.datastructures.answer import Answer
from dexter.data.datastructures.dataset import DprDataset
from dexter.data.datastructures.evidence import Evidence
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.sample import Sample
from dexter.data.loaders.BaseDataLoader import GenericDataLoader


class WikiMultihopQADataLoader(GenericDataLoader):
    '''Data loader class to load Datset from raw WikiMultihopQA dataset.
    WikiMultihopQA dataset consists of a questions each of which have a single answer and various evidences.
    
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
        self.corpus = corpus
        self.titles = [self.corpus[idx].title().split(" - ")[0] for idx,_ in enumerate(self.corpus)]
        print(self.titles[100],self.corpus[100].title())
        super().__init__(dataset, tokenizer, config_path, split, batch_size)


    def load_raw_dataset(self, split=Split.TRAIN):
        dataset = self.load_json(split)
        print(len(dataset))
        for  query_index, data in enumerate(tqdm.tqdm(dataset[:1200])):
            if len(data["context"]) == 0:
                data["context"] = ['some random title', ['some random stuff']]
            for evidence_set in data["context"]:
                title = evidence_set[0]
                #for evidence in evidence_set[1]:
                evidence = " ".join(evidence_set[1])
                #print(list(self.titles).index(title.split(" - ")[0]))
                self.raw_data.append(
                    Sample(query_index, Question(data["question"],idx=data["_id"]), Answer(data["answer"]),
                            Evidence(text=evidence, 
                                    idx=list(self.titles).index(title.split(" - ")[0]),title=title)
                ))

    def load_tokenized(self):
        if self.tokenized_path and os.path.exists(self.tokenized_path):
            self.logger.info("Loading DPR data from {}".format(self.tokenized_path))
            with open(self.tokenized_path, "r") as f:
                ip_ids, ip_attention, evidence_ids, evidence_attention = json.load(f)
        else:
            ip_ids, ip_attention = self.tokenize_questions()
            evidence_ids, evidence_attention = self.tokenize_evidences()
        return DprDataset(ip_ids, ip_attention, evidence_ids, evidence_attention)