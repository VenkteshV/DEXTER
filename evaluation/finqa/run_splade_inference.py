
from dexter.retriever.sparse.SPLADE import SPLADE
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

from dexter.config.constants import Split
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="naver/splade_v2_max",
                                     document_encoder_path="naver/splade_v2_max"
                                     ,batch_size=4)


    loader = RetrieverDataset("finqa","finqa-corpus","evaluation/config.ini",Split.DEV)
    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    tasb_search = SPLADE(config_instance)

    similarity_measure = CosineSimilarity()
    response = tasb_search.retrieve(corpus,queries,100,similarity_measure,data_name="finqa")
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))