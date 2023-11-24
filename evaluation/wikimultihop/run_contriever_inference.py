import json
from retriever.Contriever import Contriever
from data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import CosineSimilarity as CosScore
from data.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="facebook/contriever",
                                     document_encoder_path="facebook/contriever"
                                     ,batch_size=32)
   # config = config_instance.get_all_params()
    corpus_path = "/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json"

    loader = WikiMultihopQADataLoader(dataset="wikimultihopqa", config_path="evaluation/config.ini", split=Split.DEV,corpus_path=corpus_path)
    queries, qrels, corpus = loader.load_corpus_qrels_queries(Split.DEV,corpus_path)
    print("queries",len(queries),len(qrels),len(corpus),queries[0],qrels["0"])
    tasb_search = Contriever(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = CosScore()
    response = tasb_search.retrieve(corpus,queries,100,similarity_measure)
    print("indices",len(response))
    metrics = RetrievalMetrics()
    print(metrics.evaluate_retrieval(qrels=qrels,results=response,k_values=[1,10,100]))