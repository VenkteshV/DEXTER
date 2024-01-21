import json
from data.loaders.RetrieverDataset import RetrieverDataset
from methods.ir.dense.dpr.models.dpr_sentence_transformers_inference import DprSentSearch
from data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader
from constants import Split
from metrics.SimilarityMatch import CosineSimilarity as CosScore

from metrics.retrieval.RetrievalMetrics import RetrievalMetrics

from data.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":
    config_instance = DenseHyperParams(query_encoder_path="facebook-dpr-question_encoder-multiset-base",
                                     document_encoder_path="facebook-dpr-ctx_encoder-multiset-base",
                                     ann_search="faiss_search",show_progress_bar=True,
                                     batch_size=32)


    loader = RetrieverDataset("ambignq","ambignq-corpus",
                               "evaluation/config.ini", Split.DEV,tokenizer=None)     
    queries, qrels, corpus = loader.qrels()
    dpr_sent_search = DprSentSearch(config_instance)
    _ = dpr_sent_search.get_ann_algo(768, 100, "euclidean")

    dpr_sent_search.create_index(corpus)

    similarity_measure = CosScore()
    response = dpr_sent_search.retrieve_in_chunks(corpus,queries,100,
    similarity_measure,chunk=True,chunksize=400000)

    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))