#multi-qa-mpnet-base-cos-v1
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.retriever.dense.DenseFullSearch import DenseFullSearch
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="multi-qa-mpnet-base-cos-v1",
                                     document_encoder_path="multi-qa-mpnet-base-cos-v1"
                                     ,batch_size=32)
    # config = config_instance.get_all_params()
    corpus_path = "/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json"

    loader = RetrieverDataset("finqa","finqa-corpus","evaluation/config.ini",Split.DEV)
    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    tasb_search = DenseFullSearch(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = CosineSimilarity()
    response = tasb_search.retrieve(corpus,queries,100,similarity_measure)
    print("indices",len(response),response)
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))