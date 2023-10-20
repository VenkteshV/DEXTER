from typing import List
from data.datastructures.hyperparameters.base import BaseHyperParameters


class DprHyperParams(BaseHyperParameters):
    def __init__(self, query_max_length: int=64,
                 query_encoder_path: str = "facebook/dpr-question_encoder-multiset-base",
                 document_encoder_path: str = "facebook/dpr-ctx_encoder-multiset-base",
                 learning_rate: float = 1e-5,
                 num_negative_samples: int = 5,
                 ann_search: str = "faiss_search") -> None:
        super().__init__()

        self.query_max_length = query_max_length
        self.query_encoder_path = query_encoder_path
        self.document_encoder_path = document_encoder_path
        self.learning_rate = learning_rate
        self.num_negative_samples = num_negative_samples
        self.ann_search = ann_search

    def get_all_params(self):
        config = {
            "query_length" : self.query_max_length,
            "query_encoder_path": self.query_encoder_path,
            "document_encoder_path": self.document_encoder_path,
            "learning_rate": self.learning_rate,
            "num_negative_samples": self.num_negative_samples,
            "ann_search": self.ann_search
        }
        return config
