
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.config.constants import Split
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.retriever.dense.DprSentSearch import DprSentSearch
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics


if __name__ == "__main__":
    config_instance = DenseHyperParams(query_encoder_path="facebook-dpr-question_encoder-multiset-base",
                                     document_encoder_path="facebook-dpr-ctx_encoder-multiset-base",
                                     ann_search="faiss_search", convert_to_tensor=False,
                                     convert_to_numpy=True,
                                     show_progress_bar=True)

    loader = RetrieverDataset("finqa","finqa-corpus","evaluation/config.ini",Split.TEST)
    queries, qrels, corpus = loader.qrels()
    dpr_sent_search = DprSentSearch(config_instance,dataset_name="finqa")
    _ = dpr_sent_search.get_ann_algo(768, 100, "euclidean")

    dpr_sent_search.create_index(corpus)
    response = dpr_sent_search.retrieve(
        queries, 100)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))