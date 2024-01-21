import logging
from typing import List, Dict
from data.datastructures.question import Question
from data.datastructures.evidence import Evidence

from re_ranker.CrossEncoder import CrossEncoder
class Reranker:
    
    def __init__(self, model_name, batch_size: int = 128, **kwargs):
        self.cross_encoder = CrossEncoder(model_name)
        self.batch_size = batch_size
        self.rerank_results = {}
        
    def rerank(self, 
               corpus: List[Evidence], 
               queries: List[Question],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids = [], []
        corpus_final = {}
        for evidence in corpus:
            corpus_final[str(evidence.id())] = {"title":evidence.title(),"text":evidence.text()}  

        queries_final = {}
        for index, query in enumerate(queries):
             queries_final[str(query.id())] = query.text()
        # print("query_ids**",query_ids)
        #queries = [query.text() for query in list(queries)]

        for query_id in results:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus_final[doc_id].get("title", "") + " " + corpus_final[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries_final[query_id], corpus_text])

        logging.info("Intitating Rerank Top-{}....".format(top_k))
        rerank_scores = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]

        self.rerank_results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[str(query_id)][str(doc_id)] = score

        return self.rerank_results 