


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
    def retrieve(self, 
               corpus: List[Evidence], 
               queries: List[Question], 
               top_k: int, 
               return_sorted: bool = True, 
               **kwargs) -> Dict[str, Dict[str, float]]:
            corpus_texts = [(evidence.title() + self.sep + evidence.text()).strip() for evidence in corpus]
            queries_text = [question.text() for question in queries]
            with Run().context(RunConfig(nranks=1, experiment="colbert")):

                self.indexer.index(name="colbert", collection=corpus_texts, overwrite="reuse")

                self.indexer.get_index()
                searcher = Searcher(index="colbert", collection=corpus_texts)

                result_qrels = {}
                for idx,query in tqdm.tqdm(enumerate(queries)):
                    result_qrels[str(query.id())] = {}
                    results = searcher.search(query.text(), k=top_k)
                    for passage_id, passage_rank, passage_score in zip(*results):
                        result_qrels[str(query.id())][str(corpus[passage_id].id())] = passage_score
                #results = searcher.search_all(queries_text,k=100)
            return result_qrels