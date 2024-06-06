from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.FinQAMatch import FinQAMatch
if __name__=="__main__":
        config_instance = LLMEngineOrchestrator()
        llm_instance = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo")
        #assertTrue(isinstance(llm_instance, OpenAIEngine))
        with open("/home/venky/venky_bcqa/BCQA/finqa_colbert_docs_1.json") as f:
                evidence = json.load(f)
        question_df = {"questions":[],"answers":[]}


        loader = RetrieverDataset("finqa","finqa-corpus","evaluation/config.ini",Split.TEST,tokenizer=None)
    
        queries, qrels, corpus = loader.qrels()
        raw_data = loader.base_dataset.raw_data
        system_prompt = "Given the question, table and text, think step by step decompose the question and use information in table and text, output final answer for the question and give answer in form of  [Final Answer]: \n"
        matches = 0
        mismatches = 0
        ids = []
        evidences = []
        finqa_metric = FinQAMatch()
        for index,row in enumerate(raw_data):
                if row.question.id() in ids:
                        continue
                else:
                        ids.append(row.question.id())
                top_3 = " ".join(evidence[row.question.id()][0:10])
                #print(evidence_text)
                try:
                        user_prompt = """ Think step by step, use information from given table and text"""+"Table and Text:"+top_3 +"and answer the Question:""" +row.question.text()+"""Give step by step solution rationale preceded by Rationale: and  direct final answer preceded by [Final Answer]: wihtout lengthy descriptions"""

                        answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                # if "not possible" in chain_answer.lower():
                #         mismatches+=1
                #         continue
                # elif "unknown" in chain_answer.lower():
                #         mismatches+=1
                #         continue
                except:
                        top_3 = " ".join(evidence[row.question.id()][0:10])

                        user_prompt = """
        Think step by step, use information from given table and text"""+"Table and Text:"+top_3 +"and answer the Question:""" +row.question.text()
                        #print("user_prompt",user_prompt)

                        answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                if len(answer.split("[Final Answer]:")) >1:
                        if "yes" in answer.split("[Final Answer]:")[-1].lower():
                                ans = "yes"
                        elif "no" in answer.split("[Final Answer]:")[-1].lower():
                                # print(answer.split("Answer:")[1])
                                ans = "no"
                        else:
                                ans = finqa_metric.extract_num_from_str(answer.split("[Final Answer]:")[-1])
                                ans = finqa_metric._clean_num(ans)
                elif len(answer.split("answer is")) >1:
                        if answer.split("answer is")[-1] == "UNKNOWN":
                                mismatches+=1
                                continue
                else:
                        ans=finqa_metric.extract_num_from_str(answer)
                        ans = finqa_metric._clean_num(ans)
                print(row.answer.text())
                if str(row.answer.text()).lower() == "yes" or str(row.answer.text()).lower() == "no":
                        #print(index,ground_truth[index]["answer"], ans)
                        if str(row.answer.text()) == ans:
                                matches+=1
                        else:
                                mismatches+=1
                else:
                        print("******",ans, finqa_metric._clean_num(str(row.answer.text())))
                        if finqa_metric.finqa_equal(ans, finqa_metric._clean_num(str(row.answer.text())), True, True):
                                matches+=1
                        else:
                                mismatches+=1
                question_df["answers"].append(answer)
                question_df["questions"].append(str(row.question.text()))


                final_questions = pd.DataFrame(question_df)
                print("EM", matches/(matches+mismatches))
                print(final_questions)
                evidences = []
                final_questions.to_csv("chatgpt_finqa_rag_zero_shot_cot.tsv",sep="\t",index=False)


