from configparser import ConfigParser
import heapq
from typing import Dict, List, Union
import numpy as np

from torch import Tensor
import torch
from data.datastructures.evidence import Evidence
from data.datastructures.hyperparameters.dpr import DenseHyperParams
from data.datastructures.question import Question
from metrics.SimilarityMatch import SimilarityMetric
from retriever.BaseRetriever import BaseRetriver
from sentence_transformers import SentenceTransformer
import logging

from retriever.DenseFullSearch import DenseFullSearch


class ANCE(DenseFullSearch):
    #Wrapper for ANCE on top of Dense.
    # Will be different from parent when train loop implemented

    def __init__(self,config_path="config.ini",show_progress_bar=True,convert_to_tensor=True,batch_size=32) -> None:        
        self.config = ConfigParser()
        self.config.read(config_path)
        question_encoder = self.config["Retrieval"]["question-encoder"]
        context_encoder = self.config["Retrieval"]["context-encoder"]
        show_progress_bar = show_progress_bar
        convert_to_tensor = convert_to_tensor
        batch_size = batch_size
        dense_hyperparams = DenseHyperParams(query_encoder_path=question_encoder,document_encoder_path=context_encoder,show_progress_bar=show_progress_bar,convert_to_tensor=convert_to_tensor,batch_size=batch_size)        
        super().__init__(dense_hyperparams)