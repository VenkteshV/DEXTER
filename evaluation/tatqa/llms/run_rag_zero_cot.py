from llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset


if __name__=="__main__":
        config_instance = LLMEngineOrchestrator()
        llm_instance = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo")
        #assertTrue(isinstance(llm_instance, OpenAIEngine))
        with open("/home/venky/venky_bcqa/BCQA/TATQA_colbert_docs.json") as f:
                evidence = json.load(f)
        question_df = {"questions":[],"answers":[]}

        loader = RetrieverDataset("tatqa","tatqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)
        queries, qrels, corpus = loader.qrels()
        raw_data = loader.base_dataset.raw_data
        system_prompt = "Given the question, table and text think step by step decompose it and use information in table and text, output final answer for the question and give answer in form of  [Final Answer]: \n"
        matches = 0
        mismatches = 0
        ids = []
        for row in raw_data:
                if row.question.id() in ids:
                        continue
                else:
                        ids.append(row.question.id())
                top_3 = " ".join(evidence[row.question.id()][0:10])
                user_prompt = """Think step by step, use information from given table and text and give answer preceded by [Final Answer]: for the Question:"""+top_3 +row.question.text()
                print("user_prompt",user_prompt)
                chain_answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                if len(chain_answer.split("[Final Answer]:")) >1:
                        answer = chain_answer.split("[Final Answer]:")[-1]
                        print("************", answer, row.answer)
                        if type(row.answer) == list:
                                # if len(row.answer) >1:
                                #     gold_answer = "".join(row.answer)
                                # else:
                                gold_answer = row.answer[-1]
                        else:
                                gold_answer = row.answer
                        if str(gold_answer).lower().strip() in answer.lower():
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
                final_questions.to_csv("chatgpt_tatqa_rag_10_zero_cot.tsv",sep="\t",index=False)


