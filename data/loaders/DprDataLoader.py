import json
import numpy as np
import random
import uuid
from constants import Split
from data.datastructures.answer import AmbigNQAnswer
from data.datastructures.dataset import DprDataset
from data.datastructures.question import Question
from data.datastructures.evidence import Evidence

from data.datastructures.sample import Sample, AmbigNQSample
from data.loaders.Tokenizer import Tokenizer
from data.datastructures.hyperparameters.dpr import DenseHyperParams

from data.loaders.BaseDataLoader import GenericDataLoader
import logging

logger = logging.getLogger(__name__)


class DprDataLoader(GenericDataLoader):
    def __init__(self, dataset: str, tokenizer="bert-base-uncased", config_path='test_config.ini', config=None,
                 split=Split.TRAIN,
                 batch_size=None):
        super().__init__(dataset, tokenizer, config_path, config, split, batch_size)

    def load_json(self, split=Split.TRAIN):
        dataset_path = f'{self.data_folder_path}/{self.config["File-Path"]["file"]}.json'
        with open(dataset_path, 'r') as fp:
            dataset = json.load(fp)
        return dataset

    def load_dataset(self, config: DenseHyperParams, split=Split.TRAIN, shuffle: bool = False):
        dataset = self.load_json(split)
        for data in dataset:
            samples = []
            _id = data['id'] if "id" in data.keys() else uuid.uuid4()
            question = Question(data["question"], None)
            sample = {}
            evidences = {}

            evidences["positives"] = [Evidence(document["title"] + "" + document["text"], document["passage_id"]) for
                                      document in data["positive_ctxs"]]

            evidences["negatives"] = [Evidence(document["title"] + "" + document["text"], document["passage_id"]) for
                                      document in data["negative_ctxs"]]

            if "hard_negative_ctxs" in list(data.keys()) and len(data["hard_negative_ctxs"]) > 0:
                evidences["hard_negatives"] = [
                    Evidence(document["title"] + "" + document["text"], document["passage_id"]) for document in
                    data["hard_negative_ctxs"]]

            if shuffle:
                positive_ctxs = evidences["positives"]
                positive_ctx = positive_ctxs[np.random.choice(len(evidences["positive"]))]
                random.shuffle(evidences["negatives"])
                if "hard_negative_ctxs" in list(evidences.keys()):
                    random.shuffle(evidences["hard_negatives"])
            else:
                positive_ctx = [evidences["positives"][0]]
                evidences["positive_document"] = positive_ctx
            evidences["negatives"] = evidences["negatives"][:config.num_negative_samples]
            evidences["hard_negatives"] = evidences["negatives"][:config.num_negative_samples]
            for evidence in evidences["positive_document"]:
                self.raw_data.append(Sample(_id, question, AmbigNQAnswer(data["answers"]), evidence))
        ip_ids, ip_attention = self.tokenize_questions()
        evidence_ids, evidence_attention = self.tokenize_evidences()
        return DprDataset(ip_ids, ip_attention, evidence_ids, evidence_attention)
