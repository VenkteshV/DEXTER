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
import sys
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
    with open("ambigqa_colbert.json","w") as f:
        json.dump(response,f)