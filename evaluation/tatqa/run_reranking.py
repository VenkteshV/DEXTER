import json
from retriever.Contriever import Contriever
from data.loaders.RetrieverDataset import RetrieverDataset
from data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import CosineSimilarity as CosScore
from retriever.sparse.bm25 import BM25Search
from re_ranker.ReRanker import Reranker

if __name__ == "__main__":

   # config = config_instance.get_all_params()

    loader = RetrieverDataset("tatqa","tatqa-corpus","evaluation/config.ini",Split.DEV)

    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    bm25_search = BM25Search(index_name="tatqa",initialize=True)

    ## wikimultihop
    

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    response = bm25_search.retrieve(corpus,queries,100)
    metrics = RetrievalMetrics(k_values=[1,10,100])

    print(metrics.evaluate_retrieval(qrels=qrels,results=response))
    reranker = Reranker(
    "cross-encoder/quora-roberta-large"    )
    #cross-encoder/ms-marco-MiniLM-L-6-v2
    results = reranker.rerank(corpus,queries,response,100)
    #print("results",results)
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=results))