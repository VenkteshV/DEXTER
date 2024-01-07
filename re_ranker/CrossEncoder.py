from sentence_transformers.cross_encoder import CrossEncoder as CrossEnc
from typing import List, Tuple, Dict

class CrossEncoder:
    def __init__(self, model_name: str, **kwargs):
        self.model = CrossEnc(model_name, **kwargs)
    
    def predict(self, sentences: List[Tuple[str,str]], 
                batch_size: int = 32, 
                show_progress_bar: bool = True) -> List[float]:
        return self.model.predict(
            sentences=sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar)