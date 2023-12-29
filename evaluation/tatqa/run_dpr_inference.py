import json
from data.loaders.RetrieverDataset import RetrieverDataset
from methods.ir.dense.dpr.models.dpr_sentence_transformers_inference import DprSentSearch
from data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics

from data.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":
    config_instance = DenseHyperParams(query_encoder_path="facebook-dpr-question_encoder-multiset-base",
                                     document_encoder_path="facebook-dpr-ctx_encoder-multiset-base",
                                     ann_search="faiss_search",show_progress_bar=True)

    loader = RetrieverDataset("tatqa","tatqa-corpus","evaluation/config.ini",Split.DEV)
    queries, qrels, corpus = loader.qrels()
    dpr_sent_search = DprSentSearch(config_instance)
    _ = dpr_sent_search.get_ann_algo(768, 100, "euclidean")

    dpr_sent_search.create_index(corpus)
    response = dpr_sent_search.retrieve(
        queries, 100)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))