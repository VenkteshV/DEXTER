from dexter.retriever.dense.Contriever import Contriever
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity as CosScore
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="facebook/contriever",
                                     document_encoder_path="facebook/contriever"
                                     ,batch_size=32)
    # config = config_instance.get_all_params()
    corpus_path = "/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json"

    loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV)
    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    tasb_search = Contriever(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = CosScore()
    response = tasb_search.retrieve(corpus,queries,100,similarity_measure)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))