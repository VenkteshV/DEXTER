import logging
from typing import List, Dict, Tuple
def top_k_accuracy(
        qrels: Dict[str, Dict[str, int]], 
        results: Dict[str, Dict[str, float]], 
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    top_k_acc = {}
    
    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    logging.info("\n")
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = [item[0] for item in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]]
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"]/len(qrels), 5)
        logging.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    return top_k_acc