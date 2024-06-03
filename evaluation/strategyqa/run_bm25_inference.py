from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.config.constants import Split
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.retriever.lexical.bm25 import BM25Search


if __name__ == "__main__":

    # config = config_instance.get_all_params()

    loader = RetrieverDataset("strategyqa","strategyqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)

    queries, qrels, corpus = loader.qrels()
    password = "EEjHn3N6AZnHPCia-Zfm"
    cert_path = "/home/venky/numerical-llm/elasticsearch-8.11.3/config/certs/http_ca.crt"
    bm25_search = BM25Search(index_name="strategyqa",initialize=True, elastic_passoword=password,cert_path=cert_path)


    response = bm25_search.retrieve(corpus,queries,100)
    #print("indices",len(response),response,qrels)
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))