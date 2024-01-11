import json
from retriever.Contriever import Contriever
from data.loaders.RetrieverDataset import RetrieverDataset
from data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import CosineSimilarity as CosScore
from retriever.sparse.bm25 import BM25Search


if __name__ == "__main__":

   # config = config_instance.get_all_params()
    corpus_path = "/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json"

    loader = RetrieverDataset("ambignq","ambignq-corpus",
                               "evaluation/config.ini", Split.DEV,tokenizer=None)
    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    bm25_search = BM25Search(index_name="ambigqa",initialize=False)

    ## wikimultihop
    

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    response = bm25_search.retrieve(corpus,queries,100)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))