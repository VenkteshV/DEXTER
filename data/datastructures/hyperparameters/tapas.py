from typing import List
from data.datastructures.hyperparameters.base import BaseHyperParameters


class TAPASHyperParams(BaseHyperParameters):
    def __init__(self,
                 nli_tokenizer:str = "google/tapas-base-finetuned-tabfact",
                 nli_model:str = "google/tapas-base-finetuned-tabfact",
                 convert_to_tensor: bool = None, 
                 show_progress_bar: bool = None, batch_size: int = None) -> None:
        super().__init__()

        self.nli_model = nli_model,
        self.nli_tokenizer = nli_tokenizer,
        self.convert_to_tensor = convert_to_tensor
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size

    def get_all_params(self):
        config = {
            "nli_model" : self.nli_model,
            "nli_tokenizer": self.nli_tokenizer,
            "convert_to_tensor": self.convert_to_tensor,
            "batch_size": self.batch_size,
            "show_progress_bar": self.show_progress_bar
        }
        return config
