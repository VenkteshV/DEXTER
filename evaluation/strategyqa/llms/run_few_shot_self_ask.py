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


        loader = RetrieverDataset("strategyqa","strategyqa-corpus",
                               "evaluation/config.ini", Split.DEV,tokenizer=None)        
        queries, qrels, corpus = loader.qrels()
        raw_data = loader.base_dataset.raw_data
        system_prompt = "Following the given examples, Given the question think step by step decompose it and context output final answer for the question and give answer as one of true or false in form of  [Final Answer]: \n"
        matches = 0
        mismatches = 0
        ids = []
        for row in raw_data:
                if row.question.id() in ids:
                        continue
                else:
                        ids.append(row.question.id())
                user_prompt = """
                                      [Original Question]: Are any animals in Chinese calendar Chordata?
                      [Question1]: What animals are on the Chinese calendar? 
                      [Answer1]:The chinese zodiac based on the Chinese calendar has a number of animals including dogs and pigs.
                      [Question2]: What is chordata? 
                      [Answer2]: Chordata is a scientific classification of an animals phylum.
                      [Question3]: Which animals in zodiac calendar have a notochord and dorsal neural tube? Are they chordata ? 
                      [Answer3]: phylum of pigs is chordata
                      [Final Answer]: true \n\n
                                            [Original Question]: Does Andrew Johnson's presidential number exceed Elagabalus's Emperor number?,
                      [Question1]: What number president was Andrew Johnson? 
                      [Answer1]: 17,
                      [Question2]: What number emperor  was Elagabalus? 
                      [Answer2]: 25,
                      [Question3]: Is 17 greater than 25? 
                      [Answer3]: No,
                      [Final Answer]: False \n\n 
                 [Original Question]: Are more people today related to Genghis Khan than Julius Caesar?,
                      [Question1]: How many kids did Julius Caesar have? 
                      [Answer1]: 3,
                      [Question2]: How many kids did Genghis Khan have?? 
                      [Answer2]: 16,
                      [Question3]: Is 16 greater than 3? 
                      [Answer3]: Yes,
                      [Final Answer]: True \n\n 

                [Original Question]: Will the Albany in Georgia reach a hundred thousand occupants before the one in New York?,
                      [Question1]: What is the population of Albany, Georgia? 
                      [Answer1]: 75000,
                      [Question2]: What is the population of Albany, New York? 
                      [Answer2]: 100,000,
                      [Question3]: Is  75000 less than 100,000? 
                      [Answer3]: Yes,
                      [Final Answer]: False \n\n 

         [Original Question]: Would an uninsured person be more likely than an insured person to decline a CT scan?,
                      [Question1]: Typically how much does it cost to get a CT scan without insurance? 
                      [Answer1]: 5000,
                      [Question2]: Typically how much does it cost to get a CT scan with insurance?? 
                      [Answer2]: 0,
                      [Question3]: Is 5000 greater than 0? 
                      [Answer3]: Yes,
                      [Final Answer]: True \n\n 
                Following above examples, Think step by step and answer the Question:"""+row.question.text()
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
                        if str(row.answer.text()).lower() in answer.lower():
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
                final_questions.to_csv("chatgpt_strategy_self_ask.tsv",sep="\t",index=False)


