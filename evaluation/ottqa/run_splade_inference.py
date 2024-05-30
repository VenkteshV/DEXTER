import json
from retriever.sparse.SPLADE import SPLADE
from data.loaders.RetrieverDataset import RetrieverDataset
from config.constants import Split
from utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from utils.metrics.SimilarityMatch import CosineSimilarity as CosScore
from data.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="naver/splade_v2_max",
                                     document_encoder_path="naver/splade_v2_max"
                                     ,batch_size=4)


    loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)
    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    splade_search = SPLADE(config_instance)

    similarity_measure = CosScore()
    response = splade_search.retrieve(corpus,queries,100,similarity_measure,chunk=True,chunksize=200000)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))
    res = metrics.evaluate_retrieval(qrels=qrels,results=response)
    with open("evaluation/ottqa/results/ottqa_splade.json","w+") as fp:
        json.dump(res,fp) 