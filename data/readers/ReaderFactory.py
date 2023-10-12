from enum import Enum

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, AlbertConfig

from data.readers.SpanPredictors import BertSpanPredictor, AlbertSpanPredictor


class ReaderName(Enum):
    BERT = "bert"
    ALBERT = "albert"


class Reader:
    def __int__(self, span_predictor, is_train=False):
        self.model = span_predictor
        self.is_train = is_train


    def train(self):
        pass


class ReaderFactory:
    def __int__(self):
        pass

    def create_reader(self, reader_name, base, checkpoint=None):
        if reader_name == ReaderName.BERT.name:
            Config = BertConfig
            Model = BertSpanPredictor
        elif reader_name == ReaderName.ALBERT:
            Config = AlbertConfig
            Model = AlbertSpanPredictor
        else:
            raise NotImplemented(f"{reader_name} not implemented yet.")

        if checkpoint:
            model = self._load_from_checkpoint(checkpoint, base, Model, Config)
        else:
            model = Model.from_pretrained(base)
        return model

    def _load_from_checkpoint(self, checkpoint, base, Model, Config):
        # TODO: DeepaliP98GPU handling
        # def convert_to_single_gpu(state_dict):
        #     if "model_dict" in state_dict:
        #         state_dict = state_dict["model_dict"]
        #
        #     def _convert(key):
        #         if key.startswith("module."):
        #             return key[7:]
        #         return key
        #
        #     return {_convert(key): value for key, value in state_dict.items()}

        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        model = Model(Config.from_pretrained(base))
        # TODO: DeepaliP98 figure out how to incorporate the below for bart
        # if "bart" in args.bert_name:
        #     model.resize_token_embeddings(len(tokenizer))
        # logger.info("Loading from {}".format(checkpoint))
        return model.from_pretrained(None, config=model.config, state_dict=state_dict)
