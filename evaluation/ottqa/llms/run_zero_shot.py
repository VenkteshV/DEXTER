from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset


if __name__=="__main__":
        config_instance = LLMEngineOrchestrator()
        llm_instance = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo")
        #assertTrue(isinstance(llm_instance, OpenAIEngine))
        with open("/home/venky/venky_bcqa/BCQA/musique_colbert_docs.json") as f:
                evidence = json.load(f)
        question_df = {"questions":[],"answers":[]}


        loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)
    
        queries, qrels, corpus = loader.qrels()
        raw_data = loader.base_dataset.raw_data
        system_prompt = "Given the question, table and text, think step by step decompose the question and use information in table and text, output final answer for the question and give answer in form of  [Final Answer]: \n"
        matches = 0
        mismatches = 0
        ids = []
        evidences = []

        for index,row in enumerate(raw_data):
                if row.question.id() in ids and row.question.id() ==  raw_data[index+1].question.id():
                        #print(row.question.id(),row.evidences.text(),row.answer)
                        evidences.append(row.evidences.text())
                        continue
                elif row.question.id() in ids and row.question.id() !=  raw_data[index+1].question.id():
                       # print(row.question.id(),row.evidences.text(),row.answer)
                        evidences.append(row.evidences.text())
                        evidence_text = " ".join(evidences)
                elif row.question.id() not in ids and row.question.id() !=  raw_data[index+1].question.id():
                        #print(row.question.id(),row.evidences.text(),row.answer)
                        evidences.append(row.evidences.text())
                        evidence_text = " ".join(evidences)
                elif row.question.id() not in ids and row.question.id() ==  raw_data[index+1].question.id():
                        ids.append(row.question.id())
                        evidences = []
                        #print(row.question.id(),row.evidences.text(),row.answer)
                        evidences.append(row.evidences.text())
                        continue
                #print(evidence_text)
                try:
                    user_prompt = """Think step by step, use information from given table and text"""+"Table and Text:"+evidence_text +"and answer the Question:""" +row.question.text()
                    #print("user_prompt",user_prompt)
                    chain_answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                except:
                    evidences = evidences[:4]
                    evidence_text = " ".join(evidences)
                    user_prompt = """Think step by step, use information from given table and text"""+"Table and Text:"+evidence_text +"and answer the Question:""" +row.question.text()
                #print("user_prompt",user_prompt)
                chain_answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                # if "not possible" in chain_answer.lower():
                #         mismatches+=1
                #         continue
                # elif "unknown" in chain_answer.lower():
                #         mismatches+=1
                #         continue
                if len(chain_answer.split("[Final Answer]:")) >1:
                        answer = chain_answer.split("[Final Answer]:")[1]
                        print("************",answer,row.answer.text())
                        if type(row.answer)==list:
                                gold_answer = row.answer[-1].text()
                        else:
                                gold_answer = row.answer.text()
                        if str(gold_answer).lower() in str(answer).lower() or str(answer).lower() in str(gold_answer).lower():
                                matches+=1
                        else:
                                mismatches+=1
                else:
                        mismatches+=1
                question_df["answers"].append(chain_answer)
                question_df["questions"].append(str(row.question.text()))


                final_questions = pd.DataFrame(question_df)
                print("EM", matches/(matches+mismatches))
                print(final_questions)
                evidences = []
                final_questions.to_csv("chatgpt_ottqa_zero_cot.tsv",sep="\t",index=False)


