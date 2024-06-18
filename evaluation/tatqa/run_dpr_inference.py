from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.DprSentSearch import DprSentSearch
from dexter.config.constants import Split
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics

from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
import json

if __name__ == "__main__":
    config_instance = DenseHyperParams(query_encoder_path="facebook-dpr-question_encoder-multiset-base",
                                     document_encoder_path="facebook-dpr-ctx_encoder-multiset-base",
                                      convert_to_tensor=False,
                                     convert_to_numpy=True,     
                                     ann_search="faiss_search",show_progress_bar=True)

    loader = RetrieverDataset("tatqa","tatqa-corpus","evaluation/config.ini",Split.DEV)
    queries, qrels, corpus = loader.qrels()
    dpr_sent_search = DprSentSearch(config_instance,dataset_name="tatqa")
    _ = dpr_sent_search.get_ann_algo(768, 100, "euclidean")

    dpr_sent_search.create_index(corpus)
    response = dpr_sent_search.retrieve(
        queries, 100)
    print("indices", len(response))
    corpus_final = {}
    for evidence in corpus:
            corpus_final[evidence.id()] = {"evidence":evidence}
    wiki_docs = {}
    for index, key in enumerate(list(response.keys())):
        wiki_docs[key] = []
        for id in list(response[key].keys()):
            wiki_docs[key].append(corpus_final[id]["evidence"].text())
    with open("TATQA_dpr_docs.json","w") as f:
        json.dump(wiki_docs,f)
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))