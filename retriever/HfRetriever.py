from configparser import ConfigParser
import heapq
from typing import Dict, List, Union, Tuple, Any
import numpy as np
import os
import joblib
from torch import Tensor
import torch
from tqdm import tqdm
from data.datastructures.evidence import Evidence
from data.datastructures.question import Question
from metrics.SimilarityMatch import SimilarityMetric
from retriever.BaseRetriever import BaseRetriver
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging
from data.datastructures.hyperparameters.dpr import DenseHyperParams

class HfRetriever(BaseRetriver):

    def __init__(self,config=DenseHyperParams) -> None:
        super().__init__()
        self.config = config
        #print("self.config.query_encoder_path",self.config.query_encoder_path)
        self.question_tokenizer = AutoTokenizer.from_pretrained(self.config.query_encoder_path)
        self.context_tokenizer = AutoTokenizer.from_pretrained(self.config.document_encoder_path)
        self.question_encoder = AutoModel.from_pretrained(self.config.query_encoder_path)
        self.context_encoder = AutoModel.from_pretrained(self.config.document_encoder_path)
        self.question_encoder.cuda()
        self.context_encoder.cuda()
        self.batch_size = self.config.batch_size
        self.sep="[SEP]"
        self.logger = logging.getLogger(__name__)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings


    def encode_queries(self, 
                       queries: List[Question], 
                       batch_size: int = 16,
                         **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        with torch.no_grad():
            tokenized_questions = self.question_tokenizer([query.text() for query in queries], padding=True, truncation=True, return_tensors='pt').to("cuda")
            token_emb =  self.question_encoder(**tokenized_questions)
        print("token_emb",token_emb[0].shape)
        sentence_emb = self.mean_pooling(token_emb[0],tokenized_questions["attention_mask"])
        print("sentence_emb",sentence_emb.shape)
        return sentence_emb
    
    def encode_corpus(self, 
                      corpus: List[Evidence], 
                      **kwargs
                      ) -> Union[List[Tensor], np.ndarray, Tensor]:
        contexts = []
        for evidence in corpus:
            context = ""
            if evidence.title():
                context = (evidence.title() + self.sep + evidence.text()).strip()
            else:
                context = evidence.text().strip()
            contexts.append(context)
        context_embeddings = []
        index = 0
        pbar = tqdm(total = len(contexts))
        print("Starting encoding of contexts....")
        with torch.no_grad():
            while index < len(contexts):
                samples = contexts[index:index+self.batch_size]
                tokenized_contexts = self.context_tokenizer(samples, padding=True, truncation=True, return_tensors='pt').to("cuda")
                token_emb =  self.context_encoder(**tokenized_contexts)
                sentence_emb = self.mean_pooling(token_emb[0],tokenized_contexts["attention_mask"])
                context_embeddings.append(sentence_emb)
                index += self.batch_size
                pbar.update(self.batch_size)
        pbar.close()
        context_embeddings = torch.cat(context_embeddings,dim=0)
        print("context_embeddings",context_embeddings.shape)
        return context_embeddings

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

        query_embeddings = self.encode_queries(queries, batch_size=self.batch_size)
          

   
        self.logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        #self.logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
        embeddings, index_present = self.load_index_if_available()

        #TODO:Comment below for index usage
        index_present = False
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
            response[q.id()] = {}
            for index, id in enumerate(cos_scores_top_k_idx[idx]):
                response[q.id()][corpus[id].id()] = float(cos_scores_top_k_values[idx][index])
        return response
                