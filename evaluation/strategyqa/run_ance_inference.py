from constants import Split
from data.loaders.RetrieverDataset import RetrieverDataset
from metrics.SimilarityMatch import CosineSimilarity
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from retriever.ANCE import ANCE

if __name__ == "__main__":

    loader = RetrieverDataset("strategyqa","strategyqa-corpus",
                               "evaluation/config.ini", Split.DEV,tokenizer=None)
    retriever = ANCE("tests/retriever/test_config.ini")
    queries, qrels, corpus = loader.qrels()
    qrels_ret = retriever.retrieve(corpus,queries,100,CosineSimilarity(),True, chunk=True, chunksize=400000)
    print("queries",len(queries),len(corpus))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=qrels_ret))