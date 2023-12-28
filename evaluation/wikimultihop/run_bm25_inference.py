import json
from data.loaders.RetrieverDataset import RetrieverDataset
from retriever.Contriever import Contriever
from data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import CosineSimilarity as CosScore
from retriever.sparse.bm25 import BM25Search


if __name__ == "__main__":

   # config = config_instance.get_all_params()

    loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV)

    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    bm25_search = BM25Search(index_name="wikimusique",initialize=False)

    ## wikimultihop
    

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    response = bm25_search.retrieve(corpus,queries,100)
    print("indices",len(response),response,qrels)
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))