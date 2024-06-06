from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset


if __name__=="__main__":
        config_instance = LLMEngineOrchestrator()
        llm_instance = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo")
        #assertTrue(isinstance(llm_instance, OpenAIEngine))
        with open("/home/venky/venky_bcqa/BCQA/wqa_colbert_docs.json") as f:
                evidence = json.load(f)
        question_df = {"questions":[],"answers":[]}

        loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV)
        queries, qrels, corpus = loader.qrels()
        raw_data = loader.base_dataset.raw_data
        system_prompt = "Follow the given examples and Given the question and context output final answer for the question using information in the context and give answer in form of  [Final Answer]: \n"
        matches = 0
        mismatches = 0
        ids = []
        for row in raw_data:
                if row.question.id() in ids:
                        continue
                else:
                        ids.append(row.question.id())
                top_3 = " ".join(evidence[row.question.id()][0:10])
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
[Final Answer]: February 4, 1948.\n\n """ + "Follow the above example and Given the evidence, Evidence: "+top_3+" \n use the information and answer the Question:"+row.question.text()
                print("user_prompt",user_prompt)
                chain_answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                if "not possible" in chain_answer.lower():
                        mismatches+=1
                        continue
                elif "unknown" in chain_answer.lower():
                        mismatches+=1
                        continue
                elif len(chain_answer.split("[Final Answer]:")) >1:
                        answer = chain_answer.split("[Final Answer]:")[-1]
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
                final_questions.to_csv("chatgpt_wqa_rag_10_few_shot.tsv",sep="\t",index=False)


