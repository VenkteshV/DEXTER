import json
from data.loaders.RetrieverDataset import RetrieverDataset
from retriever.DenseFullSearch import DenseFullSearch
from data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import DotScore
from data.datastructures.hyperparameters.dpr import DenseHyperParams
from retriever.TCTColBERT import TCTColBERT


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="castorini/tct_colbert-v2-hnp-msmarco",
                                     document_encoder_path="castorini/tct_colbert-v2-hnp-msmarco"
                                     ,batch_size=16,show_progress_bar=True)
    loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV)
    queries, qrels, corpus = loader.qrels()
    tasb_search = TCTColBERT(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = DotScore()
    response = tasb_search.retrieve(corpus,queries,100,similarity_measure)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response,))