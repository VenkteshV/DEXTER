import json
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset



from dexter.retriever.dense.ANCE import ANCE
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics

if __name__ == "__main__":
    loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None,batch_size=32)
    retriever = ANCE("tests/retriever/test_config.ini")
    queries, qrels, corpus = loader.qrels()
    qrels_ret = retriever.retrieve(corpus,queries,100,CosineSimilarity(),chunk=True,chunksize=10000)

    print("indices",len(qrels_ret))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    res = metrics.evaluate_retrieval(qrels=qrels,results=qrels_ret)
    with open("evaluation/ottqa/results/ottqa_ance.json","w+") as fp:
        json.dump(res,fp) 