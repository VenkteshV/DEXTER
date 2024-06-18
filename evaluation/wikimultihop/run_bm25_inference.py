from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.config.constants import Split
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.retriever.lexical.bm25 import BM25Search
import os

if __name__ == "__main__":

    # config = config_instance.get_all_params()

    loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV)

    queries, qrels, corpus = loader.qrels()
    print("queries", len(queries), len(qrels), len(corpus), queries[0])
    cert_path = os.environ["ca_certs"]
    password = os.environ["http_auth"]
    bm25_search = BM25Search(index_name="wikimusique", initialize=True, elastic_passoword=password,
    cert_path=cert_path)

    ## wikimultihop
    

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    response = bm25_search.retrieve(corpus,queries,100)
    print("indices",len(response),response,qrels)
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))