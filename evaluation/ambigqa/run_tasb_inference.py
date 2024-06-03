
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.retriever.dense.DenseFullSearch import DenseFullSearch
from dexter.utils.metrics.SimilarityMatch import DotScore
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="msmarco-distilbert-base-tas-b",
                                     document_encoder_path="msmarco-distilbert-base-tas-b"
                                     ,batch_size=32, show_progress_bar=True)

    corpus_path = "/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json"

    loader = RetrieverDataset("ambignq","ambignq-corpus",
                               "evaluation/config.ini", Split.DEV,tokenizer=None) 
    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    tasb_search = DenseFullSearch(config_instance)


    similarity_measure = DotScore()
    response = tasb_search.retrieve(corpus,queries,100,similarity_measure,chunk=True,chunksize=400000)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))