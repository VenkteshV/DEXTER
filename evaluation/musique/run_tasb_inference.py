import json
from retriever.DenseFullSearch import DenseFullSearch
from data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import DotScore
from data.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="msmarco-distilbert-base-tas-b",
                                     document_encoder_path="msmarco-distilbert-base-tas-b"
                                     ,batch_size=16)
   # config = config_instance.get_all_params()
    corpus_path = "/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json"

    loader = MusiqueQADataLoader(dataset="musiqueqa", config_path="evaluation/config.ini", split=Split.DEV,corpus_path=corpus_path)
    queries, qrels, corpus = loader.load_corpus_qrels_queries(Split.DEV,corpus_path)
    print("queries",len(queries),len(qrels),len(corpus),queries[0],qrels["0"])
    tasb_search = DenseFullSearch(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = DotScore()
    response = tasb_search.retrieve(corpus,queries,100,similarity_measure)
    print("indices",len(response))
    metrics = RetrievalMetrics()
    print(metrics.evaluate_retrieval(qrels=qrels,results=response,k_values=[1,10,100]))