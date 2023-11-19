import json
from methods.ir.dense.dpr.models.dpr_sentence_transformers_inference import DprSentSearch
from data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics

<<<<<<< HEAD
from data.datastructures.hyperparameters.dpr import DenseHyperParams
=======
from data.datastructures.hyperparameters.dpr import DprHyperParams
>>>>>>> feature/retrieval


if __name__ == "__main__":

<<<<<<< HEAD
    config_instance = DenseHyperParams(query_encoder_path="facebook-dpr-question_encoder-multiset-base",
=======
    config_instance = DprHyperParams(query_encoder_path="facebook-dpr-question_encoder-multiset-base",
>>>>>>> feature/retrieval
                                     document_encoder_path="facebook-dpr-ctx_encoder-multiset-base",
                                     ann_search="faiss_search")
   # config = config_instance.get_all_params()
    corpus_path = "/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json"

    loader = MusiqueQADataLoader(dataset="musiqueqa", config_path="evaluation/config.ini", split=Split.DEV,corpus_path=corpus_path)
    queries, qrels, corpus = loader.load_corpus_qrels_queries(Split.DEV,corpus_path)
    print("queries",len(queries),len(qrels),len(corpus),queries[0],qrels["0"])
    queries_list = [query.text() for query in list(queries.values())]
    print("queries_list",queries_list[0])
    dpr_sent_search = DprSentSearch(config_instance)
    _ = dpr_sent_search.get_ann_algo(768, 100, "euclidean")

    ## wikimultihop

    with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
        corpus = json.load(f)
    dpr_sent_search.create_index(
        "", 100,corpus)
    response = dpr_sent_search.retrieve(
        queries_list, 100)
    print("indices",len(response))
    metrics = RetrievalMetrics()
    print(metrics.evaluate_retrieval(qrels=qrels,results=response,k_values=[1,10,100]))