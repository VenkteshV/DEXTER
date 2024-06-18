from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
from dexter.llms.openai_engine import OpenAIEngine
from dexter.config.constants import Split
from dexter.utils.metrics.AnswerF1 import AnswerF1
if __name__ == "__main__":
        config_instance = LLMEngineOrchestrator()
        answer_f1 = AnswerF1()
        llm_instance = config_instance.get_llm_engine(data="", llm_class="openai", model_name="gpt-3.5-turbo")
        with open("/home/venky/venky_bcqa/BCQA/musique_colbert_docs.json") as f:
                evidence = json.load(f)
        question_df = {"questions":[],"answers":[]}
        loader = RetrieverDataset("ambignq", "ambignq-corpus", "evaluation/config.ini", Split.DEV, tokenizer=None)      
        raw_data = loader.base_dataset.raw_data
        system_prompt = "Follow the examples and Given the question think step by step  decompose it and disambiguate the question and finally output a list of possible answers if applicable separated by newline \\n for the question preceded by  [Final Answer]: \n"
        ids = []
        overlap = 0
        f1=0
        for index, row in enumerate(raw_data):
                print("gt***",row.answer.flatten())
                if row.question.id() in ids:
                        continue
                else:
                        ids.append(row.question.id())
                user_prompt = """
                Question: What is youngest legal age of marriage possible in some US states when circumstances permit?
                [Final Answer]: 18 \n Nebraska ( 19 ) \n "Mississippi ( 21 ) \n \n
                Question: Who starred in barefoot in the park on broadway?
                [Final Answer]: Robert Redford \n Elizabeth Ashley \n \n
                Question: What is the airport code for abu dhabi?
                [Final Answer]: IATA : AUH \n ICAO : OMAA \n \n
                Question: What book of the bible is the ten commandments in?
                [Final Answer]: Exodus \n Deuteronomy \n \n
                Question: Who sings in what's love got to do with it movie?
                [Final Answer]: Tina Turner \n Laurence Fishburne \n \n
                Follow the above examples and given a question Think step by step and generate all possible and applicable. Start the set of answers with [Final Answer]: separated by newline for the Question:"""+row.question.text()
                print("user_prompt",user_prompt)
                chain_answer = llm_instance.get_chat_completion(user_prompt, system_prompt)
                print("chain_answer",chain_answer)
                if len(chain_answer.split("[Final Answer]:")) >0:
                        answers = chain_answer.split("[Final Answer]:")[1].split("\n")
                        print("row.answer.flatten()", len(row.answer.flatten()))
                        answer = [ans.lower() for ans in answers]
                        answer = [ans.replace("-", "").strip() for ans in answer]
                else:
                        answers = chain_answer.split("[Final Answer]:")[-1].split("\n")
                        print("row.answer.flatten()", len(row.answer.flatten()))
                        answer = [ans.lower() for ans in answers]
                        answer = [ans.replace("-","").strip() for ans in answer]
                        #answer = [ans.split() for ans in answer]

                gt = list(set([ans.lower() for ans in row.answer.flatten()]))
                print("************", answer, gt)
                f1 += answer_f1.get_f1(gt,answer)
                        #overlap+= len(list(set(answer).intersection(set(gt))))/len(row.answer.flatten())
                question_df["answers"].append(chain_answer)
                question_df["questions"].append(str(row.question.text()))


                final_questions = pd.DataFrame(question_df)
                print("F1", f1/(index+1))
                print(final_questions)
                final_questions.to_csv("chatgpt_ambigqa_few_shot.tsv",sep="\t",index=False)
        print("F1", f1/len(raw_data))



