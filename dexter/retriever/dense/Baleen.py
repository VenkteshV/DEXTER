from dexter.retriever.dense.ColBERT.baleen.condenser.condense import Condenser
from dexter.retriever.dense.ColBERT.baleen.hop_searcher import HopSearcher
from dexter.retriever.dense.ColBERT.baleen.engine import Baleen
from dexter.retriever.dense.ColBERT.colbert import Indexer
from dexter.retriever.dense.ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig
from dexter.data.datastructures.evidence import Evidence
from dexter.data.datastructures.question import Question
import os
import logging
import tqdm
from typing import List,Dict

class BaleenRetriever():
    #Wrapper class for future extensions
    def __init__(self,config=ColBERTConfig,
                 checkpoint=None, condenser_path = None) -> None:
       # super().__init__(config)
        self.config = config
        self.indexer = Indexer(config=self.config,
                               checkpoint=checkpoint)
        self.sep="[SEP]"
        self.logger = logging.getLogger(__name__)
        self.checkpointL1 = os.path.join(condenser_path, 'hover.checkpoints-v1.0/condenserL1-v1.0.dnn')
        self.checkpointL2 = os.path.join(condenser_path, 'hover.checkpoints-v1.0/condenserL2-v1.0.dnn')
    def retrieve(self, 
               corpus: List[Evidence], 
               queries: List[Question], 
               top_k: int, 
               return_sorted: bool = True, 
               index_name="colbert",
               **kwargs) -> Dict[str, Dict[str, float]]:
            corpus_texts = [(evidence.title() + self.sep + evidence.text()).strip() for evidence in corpus]
            queries_text = [question.text() for question in queries]
            with Run().context(RunConfig(nranks=1, experiment=index_name)):

                self.indexer.index(name=index_name, collection=corpus_texts, overwrite=True)

                self.indexer.get_index()
                searcher = HopSearcher(index=index_name)
                condenser = Condenser(checkpointL1=self.checkpointL1, 
                                      checkpointL2=self.checkpointL2,
                                    collectionX_path=collectionX_path,
                                    deviceL1='cuda:0', deviceL2='cuda:0')

                baleen = Baleen(collectionX_path, searcher, condenser)
                baleen.searcher.configure(nprobe=2, ncandidates=8192)
                result_qrels = {}
                for idx,query in tqdm.tqdm(enumerate(queries)):
                    result_qrels[str(query.id())] = {}
                    results = searcher.search(query.text(), k=top_k)
                    for passage_id, passage_rank, passage_score in zip(*results):
                        result_qrels[str(query.id())][str(corpus[passage_id].id())] = passage_score
                #results = searcher.search_all(queries_text,k=100)
            return result_qrels