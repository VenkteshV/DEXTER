from configparser import ConfigParser
import heapq
from typing import Dict, List, Union, Tuple, Any
import numpy as np
import os
import joblib
from torch import Tensor
import torch
from data.datastructures.evidence import Evidence
from data.datastructures.question import Question
from metrics.SimilarityMatch import SimilarityMetric
from retriever.BaseRetriever import BaseRetriver
from sentence_transformers import SentenceTransformer
import logging
from data.datastructures.hyperparameters.dpr import DenseHyperParams



class DenseFullSearch(BaseRetriver):

    def __init__(self,config=DenseHyperParams) -> None:
        super().__init__()
        self.config = config
        print("self.config.query_encoder_path",self.config.query_encoder_path)
        self.question_encoder = SentenceTransformer(self.config.query_encoder_path, device = "cuda")
        self.context_encoder = SentenceTransformer(self.config.document_encoder_path, device = "cuda")
        self.show_progress_bar = self.config.show_progress_bar
        self.convert_to_tensor = self.config.convert_to_tensor
        self.batch_size = self.config.batch_size
        self.sep="[SEP]"
        self.logger = logging.getLogger(__name__)
    
    def encode_queries(self, queries: List[Question], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self.question_encoder.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Evidence], **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        contexts = []
        for evidence in corpus:
            context = ""
            if evidence.title():
                context = (evidence.title() + self.sep + evidence.text()).strip()
            else:
                context = evidence.text().strip()
            contexts.append(context)
        return self.context_encoder.encode(contexts, batch_size=self.batch_size, **kwargs)

    def load_index_if_available(self)->Tuple[Any,bool]:
        if os.path.exists("indices/corpus/index"):
            corpus_embeddings = joblib.load("indices/corpus/index")
            index_present=True
        else:
            index_present = False
            corpus_embeddings=None
        return corpus_embeddings, index_present

    def retrieve(self, 
               corpus: List[Evidence], 
               queries: List[Question], 
               top_k: int, 
               score_function: SimilarityMetric,
               return_sorted: bool = True, 
               **kwargs) -> Dict[str, Dict[str, float]]:

            
        self.logger.info("Encoding Queries...")
        queries = [query.text() for query in list(queries.values())]
        print("queries****",queries[0])
        query_embeddings = self.encode_queries(queries, batch_size=self.batch_size)
          

   
        self.logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        #self.logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
        embeddings, index_present = self.load_index_if_available()
        if index_present:
            corpus_embeddings = embeddings
        else:
            corpus_embeddings = self.encode_corpus(corpus)
            joblib.dump(corpus_embeddings,"indices/corpus/index")

        # Compute similarites using either cosine-similarity or dot product
        cos_scores = score_function.evaluate(query_embeddings,corpus_embeddings)
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        response = {}
        for idx, q in enumerate(queries):
            response[str(idx)] = {}
            for index, id in enumerate(cos_scores_top_k_idx[idx]):
                response[str(idx)][str(id)] = float(cos_scores_top_k_values[idx][index])
        return response
                