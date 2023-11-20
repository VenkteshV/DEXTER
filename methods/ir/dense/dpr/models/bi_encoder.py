
import collections
import logging
import random
from typing import Tuple, List, Union
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn
from transformers import DPRQuestionEncoder, DPRContextEncoder
from data.datastructures.hyperparameters.dpr import DenseHyperParams

logger = logging.getLogger(__name__)


class BiEncoder(torch.nn.Module):
    """
    This trains the DPR encoders to maximize dot product 
    between queries and positive contexts.
    """

    def __init__(self, hypers: DenseHyperParams):
        super().__init__()
        self.hypers = hypers
        logger.info(
            f'BiEncoder: initializing from {hypers.query_encoder_path} and {hypers.document_encoder_path}')
        self.query_model = DPRQuestionEncoder.from_pretrained(
            hypers.query_encoder_path)
        self.document_model = DPRContextEncoder.from_pretrained(
            hypers.document_encoder_path)
        self.saved_debug = False

    def encode(self, model, 
               input_ids: torch.Tensor, 
               attention_mask: torch.Tensor):
        return model(input_ids, attention_mask)[0]

    def forward(
        self,
        input_ids_query: torch.Tensor,
        attention_mask_query: torch.Tensor,
        input_ids_document: torch.Tensor,
        attention_mask_document: torch.Tensor,
        label_indices: torch.Tensor,
        is_train: bool = True
    ):
        """
        All batches must be the same size 
        :param input_ids_query
        :param attention_mask_query
        :param input_ids_document
        :param attention_mask_document
        :param label_indices: labels
        :return: loss, accuracy
        """
        query_reps = self.encode(
            self.query_model, input_ids_query, attention_mask_query)
        document_reps = self.encode(
            self.document_model, input_ids_document, attention_mask_document)
        # (q * world_size) x (c * world_size)
        if is_train:
            dot_products = torch.matmul(query_reps, document_reps.transpose(0, 1))
            probs = F.log_softmax(dot_products, dim=1)
            loss = F.nll_loss(probs, label_indices)
            predictions = torch.max(probs, 1)[1]
            accuracy = (predictions == label_indices).sum() / \
                label_indices.shape[0]
            return loss, accuracy
        else:
            return query_reps, document_reps

    def save_models(self, save_dir: Union[str, os.PathLike]):
        """_summary_

        Args:
            save_dir (Union[str, os.PathLike]): _description_
        """        
        self.query_model.encoder.save_pretrained(
            os.path.join(save_dir, 'query_encoder'))
        self.document_model.encoder.save_pretrained(
            os.path.join(save_dir, 'document_encoder'))
