from collections import defaultdict
import configparser
import copy
import gzip
import json
import os
from typing import List
import zipfile
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from constants import Split,Dataset
from data.datastructures.answer import Answer, AmbigNQAnswer
from data.datastructures.dataset import ReaderDataset, PassageDataset, QADataset
from data.datastructures.passage import Passage
from data.datastructures.question import Question
from data.datastructures.sample import Sample, AmbigNQSample
from data.loaders.tokenizer import Tokenizer


class GenericDataLoader(DataLoader):
    def __init__(
        self,
        dataset: str,
        tokenizer: str = "bert-base-uncased",
        config_path: str = "config.ini",
        split: Split = Split.TRAIN,
        batch_size: int = None,
    ):
        self.raw_data = []
        self.meta = {}
        self.tokenizer_name = tokenizer
        self.tokenizer = Tokenizer(self.tokenizer_name)
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.is_training = split == Split.TRAIN
        datapath = self.config["Data-Path"][dataset]
        tokenized_path = f"{dataset}_{tokenizer}_tokenized"
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
        dataset_path = f"{self.data_folder_path}/{split}_light.json"
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
                Sample(data["id"], Question(data["question"], [Answer(data["answer"])]))
            )

    def tokenize_questions(self, MAX_LENGTH=32):
        op = self.tokenizer.tokenize(
            [data.question.text() for data in self.raw_data],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )
        return op["input_ids"], op["attention_mask"]

    def tokenize_answers(self):
        return self.tokenizer.tokenize([data.answer.text() for data in self.raw_data])

class PassageDataLoader(DataLoader):
    def __init__(self,
        dataset: str,
        subset_ids:List,
        tokenizer: str = "bert-base-uncased",
        config_path: str = "config.ini",
        ):
        self.subset_ids = subset_ids
        self.tokenizer_name = tokenizer     
        self.tokenizer = Tokenizer(self.tokenizer_name)
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.data_path = self.config["Data-Path"][dataset]
        self.tokenized_path = f"{dataset}_{tokenizer}_tokenized"
        tokenized_data = self._load_tokenized_data()
        self.dataset = PassageDataset(tokenized_data['input_ids'].keys(),tokenized_data['input_ids'],tokenized_data['attention_mask'])
    
    def _load_tokenized_data(self):
        if self.tokenized_path and os.path.exists(self.tokenized_path):
            self.logger.info("Loading DPR data from {}".format(self.tokenized_path))
            with open(self.tokenized_path, "r") as fp:
                return json.load(fp)
        else:
            self.raw_data = []
            with open('tests/data/passage.json','r') as fp:
                db = json.load(fp)
            #passages,titles = self.load_passage_db(self.data_path,copy.copy(self.subset_ids))
            passages,titles = db['passages'],db['titles']
            subset_ids = self.subset_ids
            if not(subset_ids):
                subset_ids = list(range(len(passages)))
            for i in range(len(subset_ids)):
                self.raw_data.append(Passage(text=passages[str(subset_ids[i])],idx=subset_ids[i],title=titles[str(subset_ids[i])]))            
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
        s_len = len(subset)

        with gzip.open(data_path, "rb") as f:
            _ = f.readline()
            offset = 0
            for line in tqdm(f):
                if offset in subset:
                    _id, passage, title = line.decode().strip().split("\t")
                    assert int(_id) - 1 == offset
                    passages[offset] = passage.lower()
                    titles[offset] = title.lower()
                    subset.remove(offset)
                    if(not(subset)):
                        break
                offset += 1
        assert s_len == len(titles) == len(passages)
        print("Loaded {} passages".format(len(passages)))
        return passages,titles 
        

class AmbigQADataLoader(GenericDataLoader):
    def __init__(
        self,
        dataset: str,
        tokenizer="bert-base-uncased",
        config_path="test_config.ini",
        split=Split.TRAIN,
        batch_size=None,
    ):
        super().__init__(dataset, tokenizer, config_path, split, batch_size)

    def load_raw_dataset(self, split=Split.TRAIN):
        dataset = self.load_json(split)
        for data in dataset:
            _id = data["id"]
            question = Question(data["question"], None)
            sample_answers = []
            for annotation in data["annotations"]:
                if annotation["type"] == "singleAnswer":
                    answers = [
                        [Answer(answer, None) for answer in annotation["answer"]]
                    ]
                elif annotation["type"] == "multipleQAs":
                    answers = [
                        [Answer(answer, None) for answer in pair["answer"]]
                        for pair in annotation["qaPairs"]
                    ]
                else:
                    raise TypeError("Unknown annotation type: ", annotation["type"])
                sample_answers.append(answers)
            self.raw_data.append(
                AmbigNQSample(_id, question, AmbigNQAnswer(sample_answers))
            )

    def tokenize_answers(self, MAX_LENGTH=20):
        # tokenize answers and make list for each tokenized answer
        samples = [sample.answer.flatten() for sample in self.raw_data]
        decoder_ids = []
        decoder_masks = []
        for sample in self.raw_data:
            sample_op_ids = []
            sample_op_attention_masks = []
            for answer in sample.answer.flatten():
                tokenized_ans = self.tokenizer.tokenize(
                    answer,
                    pad_to_max_length="bart" in self.tokenizer_name,
                    max_length=MAX_LENGTH,
                )
                sample_op_ids.append(tokenized_ans["input_ids"])
                sample_op_attention_masks.append(tokenized_ans["attention_mask"])
            decoder_ids.append(sample_op_ids)
            decoder_masks.append(sample_op_attention_masks)
        return decoder_ids, decoder_masks


class DataLoaderFactory:
    def __int__(self):
        pass

    def create_dataloader(self, dataloader_name,config_path,split,batch_size,tokenizer):
        if Dataset.AMBIGQA in dataloader_name:
            loader = AmbigQADataLoader
        else:
            raise NotImplemented(f"{dataloader_name} not implemented yet.")
        return loader(dataloader_name,tokenizer, config_path,split,batch_size)



class ReaderDataLoader(DataLoader):
    def __init__(self, dataset:str,passage_dataset:str,config_path,split:Split, batch_size=32,tokenizer="bert-base-uncased"): 
        self.split = split       
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.tokenizer_name = tokenizer
        self.tokenizer = Tokenizer(self.tokenizer_name)
        base_dataset = DataLoaderFactory().create_dataloader(dataset, config_path, self.split, batch_size,self.tokenizer_name)
        self.base_dataset = base_dataset
        retreival_out = f'{dataset}-{split}-out'.lower()
        retreival_results_path = self.config['Retrieval'][retreival_out]
        with open(retreival_results_path,"r") as fp:
            self.retreival_results = json.load(fp)[:11]
        psg_ids = [p_idx for retrieved in self.retreival_results for p_idx in retrieved]
        self.passage_dataloader = PassageDataLoader(passage_dataset,psg_ids,self.tokenizer_name,config_path)
        self.tokenized_data = self.load_tokenized_data()
        self.dataset = ReaderDataset(self.tokenized_data)

        if split==Split.TRAIN:
            sampler = RandomSampler(self.dataset)
        else:
            sampler = SequentialSampler(self.dataset)
        print("Dataset loaded of length", len(self.dataset))
        super(ReaderDataLoader, self).__init__(
            self.dataset, sampler=sampler, batch_size=batch_size
        )


    def load_tokenized_data(self,max_n_answers:int=30):
        features = defaultdict(list)
        for i, (q_input_ids, q_attention_mask, retrieved) in \
                tqdm(enumerate(zip(self.base_dataset.dataset.enc_ids, self.base_dataset.dataset.enc_mask, self.retreival_results))):
            assert len(q_input_ids)==len(q_attention_mask)==32
            q_input_ids = [in_ for in_, mask in zip(q_input_ids, q_attention_mask) if mask]
            assert 3<=len(q_input_ids)<=32
            p_input_ids = []
            p_attention_mask = []
            for p_idx in retrieved:
                input_id,att_mask= self.passage_dataloader.dataset.get_by_id(p_idx) 
                p_input_ids.append(input_id)
                p_attention_mask.append(att_mask)
            a_input_ids = [ans_ids[1:-1] for ans_ids in self.base_dataset.dataset.dec_ids[i]]
            detected_spans = []
            for _p_input_ids in p_input_ids:
                detected_spans.append([])
                for _a_input_ids in a_input_ids:
                    decoded_a_input_ids = self.base_dataset.tokenizer.decode(_a_input_ids)
                    for j in range(len(_p_input_ids)-len(_a_input_ids)+1):
                        if _p_input_ids[j:j+len(_a_input_ids)]==_a_input_ids:
                            detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+len(_a_input_ids)-1))
                        elif "albert" in "a" and \
                                _p_input_ids[j]==_a_input_ids[0] and \
                                13 in _p_input_ids[j:j+len(_a_input_ids)]:
                            k = j + len(_a_input_ids)+1
                            while k<len(_p_input_ids) and np.sum([_p_input_ids[z]!=13 for z in range(j, k)])<len(_a_input_ids):
                                k += 1
                            if decoded_a_input_ids==self.tokenizer.decode(_p_input_ids[j:k]):
                                detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+k-1))
            if self.split==Split.TRAIN:
                positives = [j for j, spans in enumerate(detected_spans) if len(spans)>0][:20]
                negatives = [j for j, spans in enumerate(detected_spans) if len(spans)==0][:50]
                if len(positives)==0:
                    continue
            else:
                positives = [j for j in range(len(detected_spans))]
                negatives = []
            for key in ["positive_input_ids", "positive_input_mask", "positive_token_type_ids",
                            "positive_start_positions", "positive_end_positions", "positive_answer_mask",
                            "negative_input_ids", "negative_input_mask", "negative_token_type_ids"]:
                features[key].append([])

            def _form_input(p_input_ids, p_attention_mask):
                assert len(p_input_ids)==len(p_attention_mask)
                assert len(p_input_ids)==128 or (len(p_input_ids)<=128 and np.sum(p_attention_mask)==len(p_attention_mask))
                if len(p_input_ids)<128:
                    p_input_ids += [self.tokenizer.tokenizer.pad_token_id for _ in range(128-len(p_input_ids))]
                    p_attention_mask += [0 for _ in range(128-len(p_attention_mask))]
                input_ids = q_input_ids + p_input_ids + [self.tokenizer.tokenizer.pad_token_id for _ in range(32-len(q_input_ids))]
                attention_mask = [1 for _ in range(len(q_input_ids))]  + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
                token_type_ids = [0 for _ in range(len(q_input_ids))] + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
                return input_ids, attention_mask, token_type_ids

            for idx in positives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["positive_input_ids"][-1].append(input_ids)
                features["positive_input_mask"][-1].append(attention_mask)
                features["positive_token_type_ids"][-1].append(token_type_ids)
                detected_span = detected_spans[idx]
                features["positive_start_positions"][-1].append(
                    [s[0] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
                features["positive_end_positions"][-1].append(
                    [s[1] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
                features["positive_answer_mask"][-1].append(
                    [1 for _ in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
            for idx in negatives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["negative_input_ids"][-1].append(input_ids)
                features["negative_input_mask"][-1].append(attention_mask)
                features["negative_token_type_ids"][-1].append(token_type_ids)
        return features


