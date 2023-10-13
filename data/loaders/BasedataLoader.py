import configparser
import json
import os
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from constants import Split
from data.datastructures.answer import Answer, AmbigNQAnswer
from data.datastructures.dataset import QADataset
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

        # max_length = 30  # TODO:parameterize this
        # answer = [sample.answer.flatten() + [''] * (max_length - len(sample.answer.flatten())) for sample in
        #           self.raw_data]  # TODO:optimize this
        #
        # op = [
        #     self.tokenizer.tokenize(_answer, return_tensors="pt", padding='max_length', max_length=100, truncation=True)
        #     for _answer in answer]  # TODO:parameterize this
        # return torch.stack([x['input_ids'] for x in op], dim=0), torch.stack([x['attention_mask'] for x in op], dim=0)


class ReaderDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32,is_training=False):
        self.dataset = dataset
        if is_training:
            sampler = RandomSampler(self.dataset)
        else:
            sampler = SequentialSampler(self.dataset)
        print("Dataset loaded of length", len(self.dataset))
        super(ReaderDataLoader, self).__init__(
            self.dataset, sampler=sampler, batch_size=batch_size
        )
