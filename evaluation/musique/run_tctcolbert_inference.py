import json
from data.loaders.RetrieverDataset import RetrieverDataset
from retriever.ColBERT.colbert.infra.config.config import ColBERTConfig
from retriever.DenseFullSearch import DenseFullSearch
from data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import DotScore
from data.datastructures.hyperparameters.dpr import DenseHyperParams
from retriever.TCTColBERT import TCTColBERT


if __name__ == "__main__":
    config_instance = ColBERTConfig(doc_maxlen=256, nbits=2, kmeans_niters=4,bsize=4, gpus=0)

    loader = RetrieverDataset("musiqueqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV)
    queries, qrels, corpus = loader.qrels()
    tasb_search = TCTColBERT(config_instance,checkpoint="colbert-ir/colbertv2.0")

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = DotScore()
    response = tasb_search.retrieve(corpus,queries,100)
    metrics = RetrievalMetrics(k_values=[1,10,100])
    #print(response)
    print("indices",len(response))
    musique_docs = {}
    with open("musique_colbert.json","w") as f:
        json.dump(response,f)
    for index, key in enumerate(list(response.keys())):
        musique_docs[key] = []
        for id in list(response[key].keys()):
            corpus_id = int(id)
            musique_docs[key].append(corpus[corpus_id].text())



    with open("musique_colbert_docs.json","w") as f:
        json.dump(musique_docs,f)
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))