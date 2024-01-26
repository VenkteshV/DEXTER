


from data.datastructures.hyperparameters.dpr import DenseHyperParams
from retriever.HfRetriever import HfRetriever
from .ColBERT.colbert import Indexer, Searcher
from .ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig
from .ColBERT.colbert.data import Queries, Collection
from data.datastructures.evidence import Evidence
from data.datastructures.question import Question
from metrics.SimilarityMatch import SimilarityMetric
import logging
import tqdm
import heapq
from typing import List,Dict


class TCTColBERT():
    #Wrapper class for future extensions
    def __init__(self,config=ColBERTConfig,
                 checkpoint=None) -> None:
       # super().__init__(config)
        self.config = config
        self.indexer = Indexer(config=self.config,
                               checkpoint=checkpoint)
        self.sep="[SEP]"
        self.logger = logging.getLogger(__name__)

    def retrieve_in_chunks(self, 
               corpus: List[Evidence], 
               queries: List[Question], 
               top_k: int, 
               return_sorted: bool = True,
                chunksize: int =200000, 
               **kwargs) -> Dict[str, Dict[str, float]]:
            print("retrieve in chunks")
            corpus_ids = [doc.id() for doc in corpus]
            corpus_texts = [(evidence.title() + self.sep + evidence.text()).strip() for evidence in corpus]
            queries_text = [question.text() for question in queries]
            query_ids = [query.id() for query in queries]
            result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
            self.results = {qid: {} for qid in query_ids}
            with Run().context(RunConfig(nranks=1, experiment="colbert")):
                batches = range(0, len(corpus), chunksize)
                print(batches)
                for batch_num, corpus_start_idx in enumerate(batches):
                    corpus_end_idx = min(corpus_start_idx + chunksize, len(corpus))

                    self.indexer.index(name="ambigqa", 
                    collection=corpus_texts[corpus_start_idx:corpus_end_idx], 
                    overwrite=True)

                    self.indexer.get_index()
                    searcher = Searcher(index="ambigqa", 
                    collection=corpus_texts[corpus_start_idx:corpus_end_idx])
                    for query_itr in range(len(queries)):
                        query_id = query_ids[query_itr] 
                        results = searcher.search(queries[query_itr].text(), k=top_k)                 
                        for sub_corpus_id, corpus_rank, score in zip(*results):
                            print("sub_corpus_id*********",len(corpus_ids),corpus_start_idx,sub_corpus_id)
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
                #results = searcher.search_all(queries_text,k=100)
            return self.results

    def retrieve(self, 
               corpus: List[Evidence], 
               queries: List[Question], 
               top_k: int, 
               return_sorted: bool = True, 
               chunk: bool = False,
               chunksize: int = 400000,
               **kwargs) -> Dict[str, Dict[str, float]]:
            print("chunk",chunk)
            if chunk:
                return self.retrieve_in_chunks(corpus,queries,top_k,return_sorted,chunksize=chunksize)

            corpus_texts = [(evidence.title() + self.sep + evidence.text()).strip() for evidence in corpus]
            queries_text = [question.text() for question in queries]
            with Run().context(RunConfig(nranks=1, experiment="colbert")):

                self.indexer.index(name="ambigqa", collection=corpus_texts, overwrite=True)

                self.indexer.get_index()
                searcher = Searcher(index="ambigqa", collection=corpus_texts)

                result_qrels = {}
                for idx,query in tqdm.tqdm(enumerate(queries)):
                    result_qrels[str(query.id())] = {}
                    results = searcher.search(query.text(), k=top_k)
                    for passage_id, passage_rank, passage_score in zip(*results):
                        result_qrels[str(query.id())][str(corpus[passage_id].id())] = passage_score
                #results = searcher.search_all(queries_text,k=100)
            return result_qrels