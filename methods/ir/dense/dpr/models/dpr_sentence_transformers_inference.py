from typing import List
from sentence_transformers import SentenceTransformer, util
from data.datastructures.evidence import Evidence
from data.datastructures.hyperparameters.dpr import DenseHyperParams
from data.datastructures.question import Question
from methods.ir.dense.dpr.indexer.indexer import AnnSearch
import gzip
import logging

logger = logging.getLogger(__name__)


class DprSentSearch():

    def __init__(self,
                 config: DenseHyperParams):
        self.query_encoder = SentenceTransformer(config.query_encoder_path, device='cuda')
        self.document_encoder = SentenceTransformer(
            config.document_encoder_path, device='cuda')
        self.documents = {}
        self.titles = {}
        self.data = []
        self.args = config
        self.ann_search = AnnSearch()
        self.ann_algo = None

    def get_passage_embeddings(self, passages:List[str] = None):
        return self.document_encoder.encode(passages,convert_to_tensor=self.args.convert_to_tensor,show_progress_bar=self.args.show_progress_bar)

    def get_ann_algo(self, emb_dim, num_trees: int = None, metric: str = None):
        self.ann_algo = self.ann_search.get_ann_instance(
            self.args.ann_search, self.data,
            emb_dim, num_trees, metric)
        return self.ann_algo

    def create_index(self, corpus = None):
        for data in list(corpus):
            self.documents[data.id()] = data.text()
            self.titles[data.id()] = data.title()
                
        self.index_mapping = list(self.documents.keys())

        passages = [self.documents[idx]+"[SEP]"+self.titles[idx]
                         for idx in self.index_mapping]
        
        index_exists = self.ann_algo.load_index_if_available()
        ##TODO: Uncomment below for index usage
        #index_exists = False
        if index_exists:
            logger.info(
                f'Index already exists. Loading {self.args.ann_search} index')
        else:
            passage_vectors = self.get_passage_embeddings(passages)
            assert len(passage_vectors)==len(self.index_mapping)
            self.ann_algo.create_index(passage_vectors)

    def retrieve(self, queries:List[Question], top_k):
        query_vector = self.query_encoder.encode([query.text() for query in queries],convert_to_tensor=self.args.convert_to_tensor,show_progress_bar=self.args.show_progress_bar)
        top_neighbours = self.ann_algo.get_top_n_neighbours(
            query_vector, top_k)
        response = {}
        for idx,q in enumerate(queries):
            response[str(q.id())] = {}
            for index, id in enumerate(top_neighbours["ids"][idx]):
                #print("top_neighbours[distances][idx]",top_neighbours["distances"][idx], index)
               # print("self.index_mapping[id]",self.index_mapping,id)
                if(id>=0):
                    response[str(q.id())][self.index_mapping[id]] = float(top_neighbours["distances"][idx][index])
        return response
