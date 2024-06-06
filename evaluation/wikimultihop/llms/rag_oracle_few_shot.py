from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


from dexter.config.constants import Split
from sentence_transformers import SentenceTransformer
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

from torch import Tensor
from typing import List,Dict

def get_top_k_similar_instances(
    sentence: str, data_emb: Tensor, data: List[Dict],
    k: int, threshold: float
) -> List[Dict]:
    """get top k neighbours for a sentence.

    Args:
        sentence (str): input
        data_emb (Tensor): corpus embeddings
        data (List[Dict]): corpus
        k (int): top_k to return
        threshold (float):

    Returns:
        List[Dict]: list of top_k data points
    """
    sent_emb = model.encode(sentence)
    # data_emb = self.get_embeddings_for_data(transfer_questions)
    print("new_emb", sent_emb.shape, data_emb.shape)
    text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
    results_sims = zip(range(len(text_sims)), text_sims)
    sorted_similarities = sorted(
        results_sims, key=lambda x: x[1], reverse=True)
    print("text_sims", sorted_similarities[:2])
    top_questions = []
    for idx, item in sorted_similarities[:k]:
        if item[0] > threshold:
            top_questions.append(list(data)[idx])
    return top_questions

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2",device="cpu")
if __name__=="__main__":
        config_instance = LLMEngineOrchestrator()
        llm_instance = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo")
        #assertTrue(isinstance(llm_instance, OpenAIEngine))
        question_df = {"questions":[],"answers":[]}

        loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV)
        queries, qrels, corpus = loader.qrels()
        raw_data = loader.base_dataset.raw_data
        system_prompt = "Follow the given examples and Given the question and context output final answer for the question using information in the context and give answer in form of  [Final Answer]: \n"
        matches = 0
        mismatches = 0
        ids = []
        evidences = []
        for index,row in enumerate(raw_data):

                if row.question.id() in ids and index+1<len(raw_data) and row.question.id() ==  raw_data[index+1].question.id():
                        print(row.question.id(),row.evidences.text(),row.answer)
                        evidences.append(row.evidences.text())
                        continue
                elif row.question.id() in ids and index+1<len(raw_data) and row.question.id() !=  raw_data[index+1].question.id():
                        print(row.question.id(),row.evidences.text(),row.answer)
                        evidences.append(row.evidences.text())

                elif row.question.id() not in ids and index+1<len(raw_data) and row.question.id() !=  raw_data[index+1].question.id():
                        print(row.question.id(),row.evidences.text(),row.answer)
                        evidences.append(row.evidences.text())

                elif row.question.id() not in ids and index+1<len(raw_data) and row.question.id() ==  raw_data[index+1].question.id():
                        ids.append(row.question.id())
                        evidences = []
                        print(row.question.id(),row.evidences.text(),row.answer)
                        evidences.append(row.evidences.text())
                        continue
                else:
                        print(row.question.id(),row.evidences.text(),row.answer)
                        evidences.append(row.evidences.text())                     
                evidence_emb = model.encode(evidences)
                evidences_final = get_top_k_similar_instances(row.question.text(),
                evidence_emb, evidences,3,0.5)
                evidence_text = " ".join(evidences)
                user_prompt = """[Question]: When does monsoon season end in the state the area code 575 is located?
[Final Answer]: mid-September.
[Question]: What is the current official currency in the country where Ineabelle Diaz is a citizen?
[Final Answer]: United States dollar.
[Question]: Where was the person who founded the American Institute of Public Opinion in 1935 born?
[Final Answer]: Jefferson.
[Question]: What language is used by the director of Tiffany Memorandum?
[Final Answer]: Italian.
[Question]: What is the sports team the person played for who scored the first touchdown in Superbowl 1?
[Final Answer]: Green Bay Packers.
[Question]: The birth country of Jayantha Ketagoda left the British Empire when?
[Final Answer]: February 4, 1948.\n\n """ + "Follow the above example, and Given the evidence, Evidence: "+evidence_text+" \n use the information and answer the Question:"+row.question.text()
                print("user_prompt",user_prompt)
                chain_answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                if "not possible" in chain_answer.lower():
                        mismatches+=1
                        continue
                elif "unknown" in chain_answer.lower():
                        mismatches+=1
                        continue
                elif len(chain_answer.split("[Final Answer]:")) >1:
                        answer = chain_answer.split("[Final Answer]:")[1]
                        print("************",answer,row.answer.text())
                        if row.answer.text().lower() in answer.lower():
                                matches+=1
                        else:
                                mismatches+=1
                else:
                        mismatches+=1
                question_df["answers"].append(chain_answer)
                question_df["questions"].append(row.question.text())


                final_questions = pd.DataFrame(question_df)
                print("EM", matches/(matches+mismatches))
                print(final_questions)
                final_questions.to_csv("chatgpt_wikimultihop_rag_oracle_few_shot.tsv",sep="\t",index=False)


