#multi-qa-mpnet-base-cos-v1

import json
from retriever.DenseFullSearch import DenseFullSearch
from data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from data.loaders.RetrieverDataset import RetrieverDataset
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import CosineSimilarity as CosScore
from data.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="multi-qa-mpnet-base-cos-v1",
                                     document_encoder_path="multi-qa-mpnet-base-cos-v1"
                                     ,batch_size=32, show_progress_bar=True)
   # config = config_instance.get_all_params()
    corpus_path = "/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json"

    loader = RetrieverDataset("strategyqa","strategyqa-corpus",
                               "evaluation/config.ini", Split.DEV,tokenizer=None)     
    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    tasb_search = DenseFullSearch(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = CosScore()
    response = tasb_search.retrieve(corpus,queries,100,similarity_measure,chunk=True,chunksize=400000)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))