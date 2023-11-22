from annoy import AnnoyIndex
import os
from torch import Tensor
import torch

from typing import List,Dict
import logging

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))



class AnnoySearch:
    def __init__(self,data: List[str]=None, num_trees=None,emb_dim=None, metric="angular"):
        self.num_trees=num_trees
        self.emb_dim=emb_dim
        self.metric = metric
        self.data = data
        self.ann = AnnoyIndex(self.emb_dim, metric=self.metric)


    def get_top_n_neighbours(self,query_vector: Tensor, top_k: int, return_distances: bool = True)->Dict:
        matches = []
        all_distances = []
        for vec in query_vector:
            top_matches = self.ann.get_nns_by_vector(vec,top_k,include_distances=return_distances)
            indices = [idx for idx in top_matches[0]]
            distances = [dist for dist in top_matches[1]]
            matches.append(indices)
            all_distances.append(distances)


        return {"ids": indices, "distances": distances}

    def load_index_if_available(self)->None:
        if os.path.exists("indices/annoy/index.annoy"):
            self.ann.load("indices/annoy/index.annoy")
            return True
        else:
            return False

    def create_index(self, passage_vectors):
        for index, embed in enumerate(passage_vectors):
            self.ann.add_item(index, embed)
        self.ann.build(self.num_trees)
        if not os.path.exists("indices/annoy"):
            os.makedirs("indices/annoy")
        self.ann.save("indices/annoy/index.annoy")

if __name__=="__main__":
    annoy_instance = AnnoySearch()
    annoy_instance.get_index(torch.zeros(5,3))