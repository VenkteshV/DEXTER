import json
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

from dexter.config.constants import Split
from dexter.retriever.dense.ColBERT.colbert.infra.config.config import ColBERTConfig
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import DotScore
from dexter.retriever.dense.TCTColBERT import TCTColBERT


if __name__ == "__main__":
    config_instance = ColBERTConfig(doc_maxlen=256, nbits=2, kmeans_niters=4,bsize=4, gpus=0)

    loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)
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
    wiki_docs = {}
    with open("wqa_colbert.json","w") as f:
        json.dump(response,f)
    for index, key in enumerate(list(response.keys())):
        wiki_docs[key] = []
        for idx in list(response[key].keys()):
            corpus_id = int(idx)
            wiki_docs[key].append(corpus[corpus_id].text())
    with open("wqa_colbert_docs.json","w") as f:
        json.dump(wiki_docs,f)
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))