import json
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.ColBERT.colbert.infra.config.config import ColBERTConfig
from dexter.retriever.dense.TCTColBERT import TCTColBERT
import sys

from dexter.utils.metrics.SimilarityMatch import DotScore
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
sys.path.insert(0, 'ColBERT/')

if __name__ == "__main__":
    config_instance = ColBERTConfig(doc_maxlen=128, nbits=2, kmeans_niters=4,bsize=8, gpus=0)

    loader = RetrieverDataset("ambignq","ambignq-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)
    print("qrels**********************")
 
    queries, qrels, corpus = loader.qrels()
    print("TCTColBERT")

    tasb_search = TCTColBERT(config_instance,checkpoint="colbert-ir/colbertv2.0")

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musiquep_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = DotScore()
    print("retrieve")
    response = tasb_search.retrieve(corpus,queries,100, True, chunk=True, chunksize=50000)

    metrics = RetrievalMetrics(k_values=[1,10,100])
    #print(response)
    print("indices",len(response))
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))
    ambigqa_docs = {}

    for index, key in enumerate(list(response.keys())):
        ambigqa_docs[key] = []
        for id in list(response[key].keys()):
            corpus_id = int(id)
            ambigqa_docs[key].append(corpus[corpus_id].text())
    with open("ambigqa_colbert_docs.json","w") as f:
        json.dump(ambigqa_docs,f)