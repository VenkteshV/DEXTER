import os
from data.loaders.RetrieverDataset import RetrieverDataset
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from retriever.sparse.bm25 import BM25Search


if __name__ == "__main__":

   # config = config_instance.get_all_params()

    loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)

    queries, qrels, corpus = loader.qrels()

    password = "<pass>"
    cert_path = "/<es home path>http_ca.crt"

    bm25_search = BM25Search(index_name="ottqa",initialize=True,elastic_passoword=password,cert_path=cert_path)


    response = bm25_search.retrieve(corpus,queries,100)
    #print("indices",len(response),response,qrels)
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))