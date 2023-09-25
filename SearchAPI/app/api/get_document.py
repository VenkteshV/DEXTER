import pandas as pd
from app.api.get_llm_response import get_expansion_llm
import json
import sentence_transformers
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers import util
import os

dir_path = os.path.abspath(os.path.dirname(__file__))


model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

with open(os.path.join(dir_path,"documents.json")) as f:
    corpus = json.load(f)

corpus_embeddings = model.encode(corpus["corpus"])
with open(os.path.join(dir_path,"persona_context.json"),"r") as f:
    contexts = json.load(f)
def get_context(persona):
    return contexts[persona]



def get_document_response(query):
    personas = ["environmentalist"]
    documents = []
    for persona in personas:
        context = get_context(persona)
        response = get_expansion_llm(query,context)
        print("respons",response)
        query_embed = model.encode(response)
        cos_scores = util.cos_sim(query_embed, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=2)
        print(top_results)
        for index in top_results[1]:
            documents.append((" ").join(corpus["corpus"][index].split()[:30]))

       
        

    return {"text":documents}
    