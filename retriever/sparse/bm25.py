from retriever.sparse.ElasticSearch import ElasticSearch
from retriever.BaseRetriever import BaseRetriver
import tqdm
import time
from typing import List, Dict
from data.datastructures.evidence import Evidence
from data.datastructures.question import Question
import os

def sleep(seconds):
    if seconds: time.sleep(seconds)
class BM25Search(BaseRetriver):
    def __init__(self, index_name: str, hostname: str = "https://localhost:9200",
                 cert_path:str="./http_cert",
                 elastic_passoword:str="password",
                 keys: Dict[str, str] = {"title": "title", "body": "text"}, 
                 language: str = "english",
                 batch_size: int = 128, timeout: int = 100, 
                 retry_on_timeout: bool = True, maxsize: int = 24, 
                 number_of_shards: int = "default", 
                 initialize: bool = True, sleep_for: int = 2):
        self.results = {}
        self.batch_size = batch_size
        self.initialize = initialize
        self.sleep_for = sleep_for
        self.config = {
            "hostname": hostname, 
            "index_name": index_name,
            "keys": keys,
            "timeout": timeout,
            "retry_on_timeout": retry_on_timeout,
            "maxsize": maxsize,
            "number_of_shards": number_of_shards,
            "language": language,
            "ca_certs":cert_path,
            "basic_auth":("elastic", elastic_passoword)

        }
        self.es = ElasticSearch(self.config)
        if self.initialize:
            self.initialise()
    
    def initialise(self):
        self.es.delete_index()
        sleep(self.sleep_for)
        self.es.create_index()
    
    def retrieve(self, corpus: List[Evidence], queries: List[Question], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        
        corpus_final = {}
        for evidence in corpus:
            corpus_final[evidence.id()] = {"title":evidence.title(),"text":evidence.text()}
        # Index the corpus within elastic-search
        # False, if the corpus has been already indexed
        if self.initialize:
            self.index(corpus_final)
            # Sleep for few seconds so that elastic-search indexes the docs properly
            sleep(self.sleep_for)
        
        #retrieve results from BM25 
        query_ids = [query.id() for index, query in enumerate(queries)]
        # print("query_ids**",query_ids)
        queries = [query.text() for query in list(queries)]

        
        for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc='progress'):
            query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
            results = self.es.lexical_multisearch(
                texts=queries[start_idx:start_idx+self.batch_size], 
                top_hits=top_k + 1) # Add 1 extra if query is present with documents
            
            for (query_id, hit) in zip(query_ids_batch, results):
                scores = {}
                for corpus_id, score in hit['hits']:
                        scores[corpus_id] = score
                self.results[str(query_id)] = scores
        
        return self.results
        
    
    def index(self, corpus: Dict[str, Dict[str, str]]):
        progress = tqdm.tqdm(unit="docs", total=len(corpus))
        # dictionary structure = {_id: {title_key: title, text_key: text}}
        dictionary = {idx: {
            self.config["keys"]["title"]: corpus[idx].get("title", None), 
            self.config["keys"]["body"]: corpus[idx].get("text", None)
            } for idx in list(corpus.keys())
        }
        self.es.bulk_add_to_index(
                generate_actions=self.es.generate_actions(
                dictionary=dictionary, update=False),
                progress=progress
                )