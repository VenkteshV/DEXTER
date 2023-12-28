import os
import faiss
import logging
from typing import List, Dict
from torch import Tensor

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))


class FaissSearch:
    def __init__(self,data: List[str]=None,emb_dim=None):
        self.emb_dim=emb_dim
        self.data = data
        self.ann = faiss.IndexFlatIP(self.emb_dim)


    def get_top_n_neighbours(self,query_vector: Tensor, top_k: int)->Dict:
        print("query_vector",query_vector.shape)
        distances, indices = self.ann.search(query_vector,top_k)
        assert indices.shape == distances.shape == (query_vector.shape[0],top_k)
        indices = indices.tolist()
        #passages = [self.data[idx] for passage_ids in indices for idx in passage_ids]
        return {"ids": indices, "distances": distances}

    def load_index_if_available(self)->None:
        if os.path.exists("indices/faiss/index_faiss"):
            self.ann = faiss.read_index("indices/faiss/index_faiss")
            return True
        else:
            return False
    def create_index(self, passage_vectors):
        self.ann.add(passage_vectors)
        if not os.path.exists("indices/faiss"):
            os.makedirs("indices/faiss")
        faiss.write_index(self.ann, "indices/faiss/index_faiss")