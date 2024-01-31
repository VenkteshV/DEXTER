import logging
import numpy as np
import torch
import heapq
from numpy import ndarray
from torch import Tensor
import os
from metrics.SimilarityMatch import SimilarityMetric
from tqdm.autonotebook import trange
from data.datastructures.hyperparameters.dpr import DenseHyperParams
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers.util import batch_to_device
import time
from typing import List, Dict, Tuple, Any, Union
import joblib
from retriever.BaseRetriever import BaseRetriver
from data.datastructures.evidence import Evidence
from data.datastructures.question import Question

logger = logging.getLogger(__name__)


class SPLADE(BaseRetriver):
    def __init__(self,config=DenseHyperParams):
        self.config = config
        #print("self.config.query_encoder_path",self.config.query_encoder_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.query_encoder_path)
        self.batch_size = self.config.batch_size
        self.sep="[SEP]"
        self.logger = logging.getLogger(__name__)
        self.model = SpladeNaver(self.config.query_encoder_path)
        self.model.eval()

    def load_index_if_available(self)->Tuple[Any,bool]:
        if os.path.exists("indices/corpus/index"):
            corpus_embeddings = joblib.load("indices/corpus/index")
            index_present=True
        else:
            index_present = False
            corpus_embeddings=None
        return corpus_embeddings, index_present
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[Question], **kwargs) -> np.ndarray:
        queries_text = [query.text() for query in queries]
        return self.model.encode_sentence_bert(self.tokenizer, queries_text, is_q=True, maxlen=self.config.query_max_length)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  out_features
    def encode_corpus(self, corpus: List[Evidence], **kwargs) -> np.ndarray:
        sentences = [(doc.title() + ' ' + doc.text()).strip() for doc in corpus]
        return self.model.encode_sentence_bert(self.tokenizer, sentences)
    def retrieve_in_chunks(self,
               corpus: List[Evidence], 
               queries: List[Question], 
               top_k: int, 
               score_function: SimilarityMetric,
               return_sorted: bool = True,
                chunksize: int =200000,
                  **kwargs  ):
        corpus_ids = [doc.id() for doc in corpus]
        query_embeddings = self.encode_queries(queries, batch_size=self.batch_size)  
        query_ids = [query.id() for query in queries]
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        self.results = {qid: {} for qid in query_ids}
        batches = range(0, len(corpus), chunksize)
        for batch_num, corpus_start_idx in enumerate(batches):
            self.logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(batches)))
            corpus_end_idx = min(corpus_start_idx + chunksize, len(corpus))

            # Encode chunk of corpus    
            sub_corpus_embeddings = self.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx]
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

        query_embeddings = self.encode_queries(queries, batch_size=self.batch_size)
        if chunk:
            results = self.retrieve_in_chunks(corpus, 
                                              queries,top_k=top_k,
                                              score_function=score_function,return_sorted=return_sorted,
                                              chunksize=chunksize)
            return results     

   
        self.logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        #self.logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
        embeddings, index_present = self.load_index_if_available()

        #TODO:Comment below for index usage
        #index_present = False
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
                
# Chunks of this code has been taken from: https://github.com/naver/splade/blob/main/beir_evaluation/models.py
# For more details, please refer to SPLADE by Thibault Formal, Benjamin Piwowarski and Stéphane Clinchant (https://arxiv.org/abs/2107.05720)
class SpladeNaver(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_path)

    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"]  # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        return torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1).values

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """helper function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def encode_sentence_bert(self, tokenizer, sentences: Union[str, List[str], List[int]],
                             batch_size: int = 4,
                             show_progress_bar: bool = None,
                             output_value: str = 'sentence_embedding',
                             convert_to_numpy: bool = True,
                             convert_to_tensor: bool = False,
                             device: str = None,
                             normalize_embeddings: bool = False,
                             maxlen: int = 64,
                             is_q: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = True

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == 'token_embeddings':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            # features = tokenizer(sentences_batch)
            # print(sentences_batch)
            features = tokenizer(sentences_batch,
                                 add_special_tokens=True,
                                 padding="longest",  # pad to max sequence length in batch
                                 truncation="only_first",  # truncates to self.max_length
                                 max_length=maxlen,
                                 return_attention_mask=True,
                                 return_tensors="pt")
            # print(features)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(**features)
                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1
                        embeddings.append(token_emb[0:last_mask_id + 1])
                else:  # Sentence embeddings
                    # embeddings = out_features[output_value]
                    embeddings = out_features
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings