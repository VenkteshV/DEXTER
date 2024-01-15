from typing import List
from data.datastructures.hyperparameters.base import BaseHyperParameters


class DenseHyperParams(BaseHyperParameters):
    def __init__(self, query_max_length: int=128,
                 query_encoder_path: str = "facebook/dpr-question_encoder-multiset-base",
                 document_encoder_path: str = "facebook/dpr-ctx_encoder-multiset-base",
                 learning_rate: float = 1e-5,
                 num_negative_samples: int = 5,
                 ann_search: str = "faiss_search",convert_to_tensor: bool = True, 
                 show_progress_bar: bool = True, batch_size: int = None) -> None:
        super().__init__()

        self.query_max_length = query_max_length
        self.query_encoder_path = query_encoder_path
        self.document_encoder_path = document_encoder_path
        self.learning_rate = learning_rate
        self.num_negative_samples = num_negative_samples
        self.ann_search = ann_search
        self.convert_to_tensor = convert_to_tensor
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size

    def get_all_params(self):
        config = {
            "query_length" : self.query_max_length,
            "query_encoder_path": self.query_encoder_path,
            "document_encoder_path": self.document_encoder_path,
            "learning_rate": self.learning_rate,
            "num_negative_samples": self.num_negative_samples,
            "ann_search": self.ann_search,
            "convert_to_tensor": self.convert_to_tensor,
            "batch_size": self.batch_size,
            "show_progress_bar": self.show_progress_bar
        }
        return config
