import json
from data.loaders.RetrieverDataset import RetrieverDataset
from retriever.ColBERT.colbert.infra.config.config import ColBERTConfig
from retriever.DenseFullSearch import DenseFullSearch
from data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from config.constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import DotScore
from data.datastructures.hyperparameters.dpr import DenseHyperParams
from retriever.dense.TCTColBERT import TCTColBERT


if __name__ == "__main__":
    config_instance = ColBERTConfig(doc_maxlen=256, nbits=2, kmeans_niters=4,bsize=4, gpus=0)

    loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)
    queries, qrels, corpus = loader.qrels()
    colbert = TCTColBERT(config_instance,checkpoint="colbert-ir/colbertv2.0")


    similarity_measure = DotScore()
    response = colbert.retrieve(corpus,queries,100)
    metrics = RetrievalMetrics(k_values=[1,10,100])

    print("indices",len(response))
    with open("evaluation/ottqa/colbert.json","w") as f:
        json.dump(response,f)
    corpus_final = {}
    for evidence in corpus:
            corpus_final[evidence.id()] = {"evidence":evidence}
    wiki_docs = {}
    for index, key in enumerate(list(response.keys())):
        wiki_docs[key] = []
        for id in list(response[key].keys()):
            wiki_docs[key].append(corpus_final[id]["evidence"].text())
    with open("OttQA_colbert_docs.json","w") as f:
        json.dump(wiki_docs,f)
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))