#multi-qa-mpnet-base-cos-v1

import json
from dexter.retriever.dense.DenseFullSearch import DenseFullSearch
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.config.constants import Split
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity as CosScore
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="multi-qa-mpnet-base-cos-v1",
                                     document_encoder_path="multi-qa-mpnet-base-cos-v1"
                                     ,batch_size=32, show_progress_bar=True)
    # config = config_instance.get_all_params()
    # corpus_path = "/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json"

    loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None,batch_size=32)    
    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    mpnet_search = DenseFullSearch(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)


    similarity_measure = CosScore()
    response = mpnet_search.retrieve(corpus,queries,100,similarity_measure,chunk=True,chunksize=200000)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    res = metrics.evaluate_retrieval(qrels=qrels,results=response)
    with open("evaluation/ottqa/results/ottqa_mpnet.json","w+") as fp:
        json.dump(res,fp) 