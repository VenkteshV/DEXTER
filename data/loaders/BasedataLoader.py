import configparser
import json
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from constants import Split
from data.datastructures.answer import Answer, AmbigNQAnswer
from data.datastructures.dataset import QaDataset
from data.datastructures.question import Question
from data.datastructures.sample import Sample, AmbigNQSample
from data.loaders.tokenizer import Tokenizer


class GenericDataLoader(DataLoader):
    def __init__(self, dataset: str, tokenizer: str = "bert-base-uncased", config_path: str = "config.ini",
                 split: Split = Split.TRAIN,
                 batch_size: int = None):
        self.raw_data = []
        self.meta = {}
        self.tokenizer = Tokenizer(tokenizer)
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.is_training = split == Split.TRAIN
        datapath = self.config["Data-Path"][dataset]
        if datapath.endswith("zip"):
            self.extract_zip_to_temp(dataset, self.config["Data-Path"][dataset])
            self.data_folder_path = dataset + "-" + "temp"
        else:
            self.data_folder_path = self.config["Data-Path"][dataset]

        self.dataset = self.load_dataset(split)
        if self.is_training:
            sampler = RandomSampler(self.dataset)
        else:
            sampler = SequentialSampler(self.dataset)
        print("Dataset loaded of length", len(self.dataset))
        super(GenericDataLoader, self).__init__(self.dataset, sampler=sampler, batch_size=batch_size)

    @staticmethod
    def extract_zip_to_temp(dataset: str, zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dataset + "-" + "temp")

    def load_json(self, split=Split.TRAIN):
        dataset_path = f'{self.data_folder_path}/{split}_light.json'
        with open(dataset_path, 'r') as fp:
            dataset = json.load(fp)
        return dataset

    def load_dataset(self, split=Split.TRAIN):

        dataset = self.load_json(split)
        for data in dataset:
            self.raw_data.append(Sample(data["id"], Question(data["question"], [Answer(data["answer"])])))
        ip_ids, ip_attention = self.tokenize_questions()
        op_ids, op_attention = self.tokenize_answers()
        return QaDataset(ip_ids, ip_attention, op_ids, op_attention, self.is_training)

    def tokenize_questions(self):
        op = self.tokenizer.tokenize([data.question.text() for data in self.raw_data], padding=True, truncation=True,
                                     return_tensors="pt")
        return op['input_ids'], op['attention_mask']

    def tokenize_answers(self):
        return self.tokenizer.tokenize([data.answer.text() for data in self.raw_data])


class AmbigQADataLoader(GenericDataLoader):
    def __init__(self, dataset: str, tokenizer="bert-base-uncased", config_path='test_config.ini', split=Split.TRAIN,
                 batch_size=None):
        super().__init__(dataset, tokenizer, config_path, split, batch_size)

    def load_dataset(self, split=Split.TRAIN):

        dataset = self.load_json(split)
        for data in dataset:
            _id = data['id']
            question = Question(data["question"], None)
            sample_answers = []
            for annotation in data["annotations"]:
                if annotation["type"] == 'singleAnswer':
                    answers = [[Answer(answer, None) for answer in annotation["answer"]]]
                elif annotation["type"] == 'multipleQAs':
                    answers = [[Answer(answer, None) for answer in pair["answer"]] for pair in annotation["qaPairs"]]
                else:
                    raise TypeError("Unknown annotation type: ", annotation["type"])
                sample_answers.append(answers)
            self.raw_data.append(AmbigNQSample(_id, question, AmbigNQAnswer(sample_answers)))
        ip_ids, ip_attention = self.tokenize_questions()
        op_ids, op_attention = self.tokenize_answers()
        return QaDataset(ip_ids, ip_attention, op_ids, op_attention, self.is_training)

    def tokenize_answers(self):
        max_length = 30  # TODO:parameterize this
        answer = [sample.answer.flatten() + [''] * (max_length - len(sample.answer.flatten())) for sample in
                  self.raw_data]  # TODO:optimize this

        op = [
            self.tokenizer.tokenize(_answer, return_tensors="pt", padding='max_length', max_length=100, truncation=True)
            for _answer in answer]  # TODO:parameterize this
        return torch.stack([x['input_ids'] for x in op], dim=0), torch.stack([x['attention_mask'] for x in op], dim=0)
