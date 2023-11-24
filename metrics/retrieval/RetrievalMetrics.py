import pytrec_eval
from typing import List, Dict, Tuple
from metrics.retrieval.accuracy import top_k_accuracy
import logging

logger = logging.getLogger(__name__)

class RetrievalMetrics:
    """retrieval metrics ndcg, mrr etc.
    """    
    def __init__(self,  k_values: List[int] = [1,3,5,10,100]):
        self.k_values = k_values
        self.top_k = max(k_values)

    def evaluate_retrieval(self,qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]]):
        """_summary_

        Args:
            qrels (Dict[str, Dict[str, int]]): query doc relevance mapping for evaluation of results
            results (Dict[str, Dict[str, float]]): _description_

        Returns:
            Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]: _description_
        """        


        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        accuracy = {}
        
        for k in self.k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
            accuracy[f"acc@{k}"] = 0.0
        
        map_string =  set(["map_cut."+str(k) for k in self.k_values])
        ndcg_string = set(["ndcg_cut."+str(k) for k in self.k_values])
        recall_string =  set(["recall."+str(k) for k in self.k_values])
        precision_string =  set(["P."+str(k) for k in self.k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, map_string|ndcg_string|recall_string|precision_string)
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in self.k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        
        for k in self.k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        
        for eval in [ndcg, _map, recall, precision]:
            logger.info("\n")
            for k in eval.keys():
                logger.info("{}: {:.4f}".format(k, eval[k]))
        accuracy[f"acc@{k}"] = top_k_accuracy(qrels, results, self.k_values)

        return ndcg, _map, recall, precision