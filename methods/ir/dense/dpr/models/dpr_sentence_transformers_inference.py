from sentence_transformers import SentenceTransformer, util
from data.datastructures.hyperparameters.dpr import DprHyperParams
from methods.ir.dense.dpr.indexer.indexer import AnnSearch
import gzip
import logging

logger = logging.getLogger(__name__)


class DprSentSearch():
    def __init__(self,
                 config: DprHyperParams):
        self.query_encoder = SentenceTransformer(config.query_encoder_path, device='cuda')
        self.document_encoder = SentenceTransformer(
            config.document_encoder_path, device='cuda')
        self.documents = {}
        self.titles = {}
        self.data = []
        self.args = config
        self.ann_search = AnnSearch()
        self.ann_algo = None

    def get_passage_embeddings(self, data_path, subset: int = None, corpus = None):
        if len(corpus)==0:
            with gzip.open(data_path, "rb") as f:
                _ = f.readline()
                index = 0
                for line in f:
                    if subset is not None and index >= subset:
                        break
                    else:
                        id, document, title = line.decode().strip().split("\t")
                        self.documents[index] = document
                        self.titles[index] = title
                        index += 1
        else:
            for idx in list(corpus.keys()):
                self.documents[int(idx)] = corpus[idx]["text"]
                self.titles[int(idx)] = corpus[idx]["title"]
            print("here****",len(self.documents),self.documents[0])

        self.data = [self.documents[idx]+"[SEP]"+self.titles[idx]
                         for idx in list(self.documents.keys())]
        return self.document_encoder.encode(self.data)

    def get_ann_algo(self, emb_dim, num_trees: int = None, metric: str = None):
        self.ann_algo = self.ann_search.get_ann_instance(
            self.args.ann_search, self.data,
            emb_dim, num_trees, metric)
        return self.ann_algo

    def create_index(self, data_path, subset: int = 100000, corpus = None):
        index_exists = self.ann_algo.load_index_if_available()
        if index_exists:
            logger.info(
                f'Index already exists. Loading {self.args.ann_search} index')
        else:
            passage_vectors = self.get_passage_embeddings(data_path, subset, corpus)
            self.ann_algo.create_index(passage_vectors)

    def retrieve(self, query, top_k):
        query_vector = self.query_encoder.encode(query)
        top_neighbours = self.ann_algo.get_top_n_neighbours(
            query_vector, top_k)
        #print("top_neighbours",top_neighbours)
        response = {}
        for idx, q in enumerate(query):
            response[str(idx)] = {}
            for index, id in enumerate(top_neighbours["ids"][idx]):
                response[str(idx)][str(id)] = float(top_neighbours["distances"][idx][index])
        return response
