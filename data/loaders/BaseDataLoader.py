import configparser
import copy
import gzip
import json
import os
from typing import List
import zipfile

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from constants import DataTypes, Separators, Split
from data.datastructures.answer import Answer
from data.datastructures.dataset import PassageDataset, QADataset
from data.datastructures.evidence import Evidence, TableEvidence
from data.datastructures.question import Question
from data.datastructures.sample import Sample
from data.loaders.Tokenizer import Tokenizer


class GenericDataLoader(DataLoader):
    def __init__(
        self,
        dataset: str,
        tokenizer: str = "bert-base-uncased",
        config_path: str = "config.ini",
        split: Split = Split.TRAIN,
        prefix:str = None,
        batch_size: int = None,
    ):
        self.raw_data:List[Sample] = []
        self.meta = {}
        self.tokenizer_name = tokenizer
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.is_training = split == Split.TRAIN
        print(self.config["Data-Path"].keys())

        datapath = self.config["Data-Path"][dataset]
        tokenized_path = f"{dataset}_{tokenizer}_tokenized"
        self.prefix = prefix if prefix else ""
        self.tokenized_path = (
            self.config["Data-Path"][tokenized_path]
            if tokenized_path in self.config["Data-Path"].keys()
            else None
        )
        if datapath.endswith("zip"):
            self.extract_zip_to_temp(dataset, self.config["Data-Path"][dataset])
            self.data_folder_path = dataset + "-" + "temp"
        else:
            self.data_folder_path = self.config["Data-Path"][dataset]

        self.load_raw_dataset(split)
        #If no tokenizer given, only load raw dataset
        if(self.tokenizer_name):
            self.tokenizer = Tokenizer(self.tokenizer_name)
            self.dataset = self.load_tokenized()
            if self.is_training:
                sampler = RandomSampler(self.dataset)
            else:
                sampler = SequentialSampler(self.dataset)
            print("Dataset loaded of length", len(self.dataset))
            super(GenericDataLoader, self).__init__(
                self.dataset, sampler=sampler, batch_size=batch_size
            )

    @staticmethod
    def extract_zip_to_temp(dataset: str, zip_file_path):
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(dataset + "-" + "temp")

    def load_json(self, split=Split.TRAIN):
        dataset_path = f"{self.data_folder_path}/{split}.json"
        with open(dataset_path, "r") as fp:
            dataset = json.load(fp)
        return dataset

    def load_tokenized(self):
        if self.tokenized_path and os.path.exists(self.tokenized_path):
            self.logger.info("Loading DPR data from {}".format(self.tokenized_path))
            with open(self.tokenized_path, "r") as f:
                ip_ids, ip_attention, op_ids, op_attention = json.load(f)
        else:
            ip_ids, ip_attention = self.tokenize_questions()
            op_ids, op_attention = self.tokenize_answers()
        return QADataset(ip_ids, ip_attention, op_ids, op_attention, self.is_training)

    def load_raw_dataset(self, split=Split.TRAIN):
        dataset = self.load_json(split)
        for data in dataset:
            self.raw_data.append(
                Sample(data["id"], Question(data["question"]), Answer(data["answer"])))

    def tokenize_questions(self, MAX_LENGTH=32):
        op = self.tokenizer.tokenize(
            [self.tokenizer.prefix+data.question.text() for data in self.raw_data],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )
        return op["input_ids"], op["attention_mask"]
    
    def tokenize_evidences(self):
        op = self.tokenizer.tokenize([data.evidences.text() for data in self.raw_data], padding=True, truncation=True,
                                     return_tensors="pt")
        return op['input_ids'], op['attention_mask']
        

    def tokenize_answers(self):
        return self.tokenizer.tokenize([self.tokenizer.prefix+data.answer.text() for data in self.raw_data])

class PassageDataLoader(DataLoader):
    def __init__(self,
        dataset: str,
        subset_ids:List,
        tokenizer: str = "bert-base-uncased",
        config_path: str = "config.ini",
        ):
        self.raw_data:List[Evidence] = []
        self.subset_ids = subset_ids
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.data_path = self.config["Data-Path"][dataset]  
        self.tokenizer_name = tokenizer      
        self._load_data()

        if(self.tokenizer_name):
            self.tokenized_path = f"{dataset}_{tokenizer}_tokenized"
            self.tokenizer_name = tokenizer     
            self.tokenizer = Tokenizer(self.tokenizer_name)            
            tokenized_data = self._load_tokenized_data()
            self.dataset = PassageDataset(tokenized_data['input_ids'].keys(),tokenized_data['input_ids'],tokenized_data['attention_mask'])
    
    def _load_data(self):
        if(".json" in self.data_path):
            with open(self.data_path,'r') as fp:
                db = json.load(fp) #format {"id":{"passage":"..","title":...}
                passages = {}
                titles = {}
                types = {}
                for id in tqdm(db.keys(),total=len(db),desc="Loading passages"):
                    if "text" in db[id].keys():
                        passages[id] = db[id]["text"]
                    else:
                        passages[id] = db[id]["passage"]
                    titles[id] = db[id]["title"] if db[id]["title"] else ""
                    if("type" in db[id].keys()):
                        types[id] = db[id]["type"]
        else:
            passages,titles = self.load_passage_db(self.data_path,copy.copy(self.subset_ids))
        subset_ids = self.subset_ids
        if not(subset_ids):
            subset_ids = list(passages.keys())
        for i in tqdm(range(len(subset_ids)),total = len(subset_ids),desc="Transforming passage dataset"):
            idx = str(subset_ids[i])
            if(idx in types.keys() and types[idx]==DataTypes.TABLE):
                rows = passages[idx].split(Separators.TABLE_ROW_SEP)
                columns = rows[0].split(Separators.TABLE_COL_SEP)
                table = [row.split(Separators.TABLE_COL_SEP) for row in rows[1:]]
                self.raw_data.append(TableEvidence(idx=idx,title=titles[idx],columns=columns,table=table))
            else:
                self.raw_data.append(Evidence(text=passages[str(subset_ids[i])],idx=subset_ids[i],title=titles[str(subset_ids[i])])) 

    
    def _load_tokenized_data(self):
        if self.tokenized_path and os.path.exists(self.tokenized_path):
            self.logger.info("Loading DPR data from {}".format(self.tokenized_path))
            with open(self.tokenized_path, "r") as fp:
                return json.load(fp)
        else:             
            input_data = [passage.title() + " " + "[SEP]" + " " + passage.text() for passage in self.raw_data]
            psg_ids = [passage.id() for passage in self.raw_data]
            tokenized_data = self.tokenizer.tokenize(input_data,
                            max_length=128,
                            pad_to_max_length=True)
            input_ids = {_id: _input_ids
                            for _id, _input_ids in zip(psg_ids, tokenized_data["input_ids"])}
            attention_mask = {_id: _attention_mask
                                for _id, _attention_mask in zip(psg_ids, tokenized_data["attention_mask"])}
            return {"input_ids": input_ids, "attention_mask": attention_mask}            

    
    def load_passage_db(self,data_path, subset=None):
        passages = {}
        titles = {}
        load_all = True if not(subset) else False
        s_len = len(subset) if not(load_all) else None

        with gzip.open(data_path, "rb") as f:
            _ = f.readline()
            offset = 0
            for line in tqdm(f):
                if load_all or offset in subset:
                    _id, passage, title = line.decode().strip().split("\t")
                    assert int(_id) - 1 == offset
                    passages[offset] = passage.lower()
                    titles[offset] = title.lower()
                    subset.remove(offset)
                    if(not(subset)):
                        break
                offset += 1
        if(not(load_all)):
            assert s_len == len(titles) == len(passages)
        print("Loaded {} passages".format(len(passages)))
        return passages,titles 
        

    



