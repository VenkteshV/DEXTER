import json
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.DenseFullSearch import DenseFullSearch
from dexter.config.constants import Split
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import DotScore
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":


    config_instance = DenseHyperParams(query_encoder_path="msmarco-distilbert-base-tas-b",
                                     document_encoder_path="msmarco-distilbert-base-tas-b"
                                     ,batch_size=16,show_progress_bar=True)
    loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)
    queries, qrels, corpus = loader.qrels()
    tasb_search = DenseFullSearch(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = DotScore()
    response = tasb_search.retrieve(corpus,queries,100,similarity_measure,chunk=True,chunksize=100000)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    res = metrics.evaluate_retrieval(qrels=qrels,results=response)
    with open("evaluation/ottqa/results/ottqa_tasb.json","w+") as fp:
        json.dump(res,fp) 