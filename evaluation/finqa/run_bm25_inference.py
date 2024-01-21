from data.loaders.RetrieverDataset import RetrieverDataset
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from retriever.sparse.bm25 import BM25Search


if __name__ == "__main__":

   # config = config_instance.get_all_params()

    loader = RetrieverDataset("finqa","finqa-corpus","tests/retriever/test_config.ini",Split.DEV)

    queries, qrels, corpus = loader.qrels()

    bm25_search = BM25Search(index_name="finqa",initialize=True)


    response = bm25_search.retrieve(corpus,queries,100)
    print("indices",len(response), len(queries),len(corpus))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))