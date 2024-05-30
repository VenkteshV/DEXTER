import json
from retriever.Contriever import Contriever
from data.loaders.RetrieverDataset import RetrieverDataset
from retriever.ColBERT.colbert.infra.config.config import ColBERTConfig
from retriever.DenseFullSearch import DenseFullSearch
from config.constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import CosineSimilarity as CosScore
from retriever.sparse.bm25 import BM25Search
from re_ranker.ReRanker import Reranker
from retriever.dense.TCTColBERT import TCTColBERT

if __name__ == "__main__":

   # config = config_instance.get_all_params()

    loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)

    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0],corpus[0])
    bm25_search = BM25Search(index_name="ottqa",initialize=True)

    ## wikimultihop
    

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    response = bm25_search.retrieve(corpus,queries,100)
    metrics = RetrievalMetrics(k_values=[1, 10, 100])
    config_instance = ColBERTConfig(doc_maxlen=256, nbits=2, kmeans_niters=4,bsize=8, gpus=0)
    colbert_search = TCTColBERT(config_instance,checkpoint="colbert-ir/colbertv2.0")

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musiquep_corpus.json") as f:
    #     corpus = json.load(f)
    corpus_final = {}
    for evidence in corpus:
            corpus_final[evidence.id()] = {"evidence":evidence}
    corpus_data = []
    for index, key in enumerate(list(response.keys())):
        for id in list(response[key].keys()):
                    corpus_data.append(corpus_final[id]["evidence"])
    #similarity_measure = DotScore()
    print("retrieve")
    response = colbert_search.retrieve(corpus_data,queries,100, True, chunk=True, chunksize=50000)
    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
    print("indices",len(response))
    wiki_docs = {}
    for index, key in enumerate(list(response.keys())):
        wiki_docs[key] = []
        for id in list(response[key].keys()):
            wiki_docs[key].append(corpus_final[id]["evidence"].text())
    with open("OttQA_colbert_docs.json","w") as f:
        json.dump(wiki_docs,f)
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))
    with open("ottqa_colbert_docs.json","w") as f:
        json.dump(wiki_docs,f)
    # reranker = Reranker(
    #   "cross-encoder/ms-marco-MiniLM-L-12-v2"
    # )
    # #cross-encoder/ms-marco-MiniLM-L-6-v2
    # results = reranker.rerank(corpus,queries,response,100)
    # #print("results",results)
    # metrics = RetrievalMetrics(k_values=[1,10,100])
    # print(metrics.evaluate_retrieval(qrels=qrels,results=results))