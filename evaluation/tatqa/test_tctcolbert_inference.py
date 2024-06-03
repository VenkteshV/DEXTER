from dexter.data.loaders.RetrieverDataset import RetrieverDataset
import json
from dexter.config.constants import Split
from dexter.retriever.dense.ColBERT.colbert.infra.config.config import ColBERTConfig
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import DotScore
from dexter.retriever.dense.TCTColBERT import TCTColBERT


if __name__ == "__main__":
    config_instance = ColBERTConfig(doc_maxlen=256, nbits=2, kmeans_niters=4,bsize=4, gpus=1)

    loader = RetrieverDataset("tatqa","tatqa-corpus","evaluation/config.ini",Split.DEV)
    queries, qrels, corpus = loader.qrels()
    tasb_search = TCTColBERT(config_instance,checkpoint="colbert-ir/colbertv2.0")

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = DotScore()
    response = tasb_search.retrieve(corpus,queries,100)
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print("indices", len(response))
    corpus_final = {}
    for evidence in corpus:
            corpus_final[evidence.id()] = {"evidence":evidence}
    wiki_docs = {}
    for index, key in enumerate(list(response.keys())):
        wiki_docs[key] = []
        for id in list(response[key].keys()):
            wiki_docs[key].append(corpus_final[id]["evidence"].text())
    with open("TATQA_colbert_docs.json","w") as f:
        json.dump(wiki_docs,f)
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))