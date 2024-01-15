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
        queries = [query.text() for query in queries]
        return self.question_encoder.encode(queries, batch_size=batch_size,**kwargs)
    
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

    def retrieve_in_chunks(self,
               corpus: List[Evidence], 
               queries: List[Question], 
               top_k: int, 
               score_function: SimilarityMetric,
               return_sorted: bool = True,
                chunksize: int =200000,
                  **kwargs  ):
        corpus_ids = [doc.id() for doc in corpus]
        query_embeddings = self.encode_queries(queries, batch_size=self.batch_size,show_progress_bar=self.show_progress_bar,convert_to_tensor=self.convert_to_tensor,**kwargs)  
        query_ids = [query.id() for query in queries]
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        self.results = {qid: {} for qid in query_ids}
        batches = range(0, len(corpus), chunksize)
        for batch_num, corpus_start_idx in enumerate(batches):
            self.logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(batches)))
            corpus_end_idx = min(corpus_start_idx + chunksize, len(corpus))

            # Encode chunk of corpus    
            sub_corpus_embeddings = self.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
                )

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = score_function.evaluate(query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]                  
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
        return self.results
    def retrieve(self, 
               corpus: List[Evidence], 
               queries: List[Question], 
               top_k: int, 
               score_function: SimilarityMetric,
               return_sorted: bool = True, 
               chunk: bool = False,
               chunksize = None,
               **kwargs) -> Dict[str, Dict[str, float]]:

            
        self.logger.info("Encoding Queries...")

        query_embeddings = self.encode_queries(queries, batch_size=self.batch_size,show_progress_bar=self.show_progress_bar,convert_to_tensor=self.convert_to_tensor,**kwargs)  
        self.logger.info("Encoding Corpus in batches... Warning: This might take a while!")

        if chunk:
            results = self.retrieve_in_chunks(corpus, 
                                              queries,top_k=top_k,
                                              score_function=score_function,return_sorted=return_sorted,
                                              chunksize=chunksize)
            return results
        #self.logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
        embeddings, index_present = self.load_index_if_available()
        #TODO: Comment below for index usage
        #index_present = False
        if index_present:
            corpus_embeddings = embeddings
        else:
            corpus_embeddings = self.encode_corpus(corpus,show_progress_bar=self.show_progress_bar,convert_to_tensor=self.convert_to_tensor,**kwargs)
            joblib.dump(corpus_embeddings,"indices/corpus/index")
        # Compute similarites using either cosine-similarity or dot product
        cos_scores = score_function.evaluate(query_embeddings,corpus_embeddings)
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[0])), dim=1, largest=True, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        response = {}
        for idx, q in enumerate(queries):
            response[q.id()] = {}
            for index, id in enumerate(cos_scores_top_k_idx[idx]):
                document_id = corpus[id].id()
                response[q.id()][document_id] = float(cos_scores_top_k_values[idx][index])
        return response
                