from config.constants import Split
from data.loaders.RetrieverDataset import RetrieverDataset
from metrics.SimilarityMatch import CosineSimilarity
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from methods.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
from methods.llms.openai_engine import OpenAIEngine
from config.constants import Split

if __name__ == "__main__":
        config_instance = LLMEngineOrchestrator()
        llm_instance = config_instance.get_llm_engine(data="", llm_class="openai", model_name="gpt-3.5-turbo")
        with open("/home/venky/venky_bcqa/BCQA/musique_colbert_docs.json") as f:
                evidence = json.load(f)
        question_df = {"questions":[],"answers":[]}
        loader = RetrieverDataset("ambignq", "ambignq-corpus", "evaluation/config.ini", Split.DEV, tokenizer=None)      
        raw_data = loader.base_dataset.raw_data
        system_prompt = "Given the question think step by step  decompose it and disambiguate the question and finally output a list of possible answers if applicable separated by newline \\n for the question preceded by  [Final Answer]: \n"
        ids = []
        overlap = 0
        for index, row in enumerate(raw_data):
                print("gt***",row.answer.flatten())
                if row.question.id() in ids:
                        continue
                else:
                        ids.append(row.question.id())
                user_prompt = """Think step by step and list all possible answers for the Question:"""+row.question.text()
                print("user_prompt",user_prompt)
                chain_answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                if "not possible" in chain_answer.lower():
                        #mismatches+=1
                        continue
                elif "unknown" in chain_answer.lower():
                        #mismatches+=1
                        continue
                elif len(chain_answer.split("[Final Answer]:")) >1:
                        answer = chain_answer.split("[Final Answer]:")[1].split("\n")
                        print("row.answer.flatten()", len(row.answer.flatten()))
                        answer = [ans.lower() for ans in answer]
                        answer = [ans.replace("-","") for ans in answer]
                        #answer = [ans.split() for ans in answer]

                        gt = [ans.lower() for ans in row.answer.flatten()]
                        print("************", answer, gt)

                        overlap+= len(list(set(answer).intersection(set(gt))))/len(row.answer.flatten())
                question_df["answers"].append(chain_answer)
                question_df["questions"].append(str(row.question.text()))


                final_questions = pd.DataFrame(question_df)
                print("EM", overlap/len(raw_data))
                print(final_questions)
                final_questions.to_csv("chatgpt_ambigqa_zero_cot.tsv",sep="\t",index=False)



