from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.retriever.dense.ANCE import ANCE

if __name__ == "__main__":
    loader = RetrieverDataset("tatqa","tatqa-corpus","evaluation/config.ini",Split.DEV)
    retriever = ANCE("evaluation/config.ini")
    queries, qrels, corpus = loader.qrels()
    qrels_ret = retriever.retrieve(corpus,queries,100,CosineSimilarity(),True)

    print("indices",len(queries),len(corpus),len(qrels_ret))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=qrels_ret))