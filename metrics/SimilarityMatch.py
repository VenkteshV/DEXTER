
import torch
from metrics.MetricsBase import Metric

class SimilarityMetric(Metric):

    def name(self):
        return "None"  

    def normalize_embedding(self,embeddings):
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
        return embeddings

    def evaluate(self,embeddings1,embeddings2):
        embeddings1 = self.normalize_embedding(embeddings1)
        embeddings2 = self.normalize_embedding(embeddings2)
        scores = self.score(embeddings1,embeddings2)
        scores[torch.isnan(scores)] = -1
        return scores

    def score(self,embeddings1,embeddings2):
        return None





class CosineSimilarity(SimilarityMetric):

    def name(self):
        return "Cosine Similarity"  
    
    def score(self,embeddings1,embeddings2):
        embeddings1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        scores = torch.mm(embeddings1_norm, embeddings2_norm.transpose(0, 1))
        return scores

    
class DotScore(SimilarityMetric):
    def name(self):
        return "Dot Similarity"
    
    def score(self,embeddings1, embeddings2):
        scores  = torch.mm(embeddings1, embeddings2.transpose(0, 1))
        scores[torch.isnan(scores)] = -1
        return scores
    