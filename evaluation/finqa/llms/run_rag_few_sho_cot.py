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
                top_3 = " ".join(evidence[row.question.id()][0:5])
                #print(evidence_text)
                try:
                        user_prompt = """ Follow the examples, Think step by step, use information from given table and text"""+"Table and Text:"+top_3 +"and answer the Question:""" +row.question.text()+"""Give step by step solution rationale preceded by Rationale: and  direct final answer preceded by [Final Answer]: wihtout lengthy descriptions"""

                        answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                # if "not possible" in chain_answer.lower():
                #         mismatches+=1
                #         continue
                # elif "unknown" in chain_answer.lower():
                #         mismatches+=1
                #         continue
                except:
                        top_3 = " ".join(evidence[row.question.id()][0:5])

                        user_prompt = """ 
                                              Table: $ in millions | year ended december 2014 | year ended december 2013 | year ended december 2012
fixed income currency and commodities client execution | $ 8461 | $ 8651 | $ 9914
equities client execution1 | 2079 | 2594 | 3171
commissions and fees | 3153 | 3103 | 3053
securities services | 1504 | 1373 | 1986
total equities | 6736 | 7070 | 8210
total net revenues | 15197 | 15721 | 18124
operating expenses | 10880 | 11792 | 12490
pre-tax earnings | $ 4317 | $ 3929 | $ 5634
Question: what was the percentage change in pre-tax earnings for the institutional client services segment between 2012 and 2013?
Rationale: The pre-tax earnings for the institutional client services segment in 2012 was $ 5634 million , and in 2013 was $ 3929 million. The net change in pre-tax earnings was $ 1705 million, and the percentage change was 30.3%. So the answer is 30.3%.
[Final Answer]: -30.3% \n


Read the following text and table, and then answer a question
Text: during the year ended march 31 , 2012 , the company has recorded $ 3.3 million in stock-based compensation expense for equity awards in which the prescribed performance milestones have been achieved or are probable of being achieved .
Table: - | number of shares ( in thousands ) | weighted average grant date fair value ( per share )
restricted stock and restricted stock units at beginning of year | 407 | $ 9.84
granted | 607 | 18.13
vested | -134 ( 134 ) | 10.88
forfeited | -9 ( 9 ) | 13.72
restricted stock and restricted stock units at end of year | 871 | $ 15.76
Question: during the 2012 year , did the equity awards in which the prescribed performance milestones were achieved exceed the equity award compensation expense for equity granted during the year?
Rationale: The prescribed performance milestones is 3.3 million. The number of shares is 607 thousand. The fair value is 18.13. The compoensation expense is 607 thousand * 18.13 = 11 million. So the answer is no.
[Final Answer]: no \n


Read the following text and table, and then answer a question
Text: annual sales of printing papers and graphic arts supplies and equipment totaled $ 3.5 billion in 2012 compared with $ 4.0 billion in 2011 and $ 4.2 billion in 2010 , reflecting declining demand and the exiting of unprofitable businesses .
Table: in millions | 2012 | 2011 | 2010
sales | $ 6040 | $ 6630 | $ 6735
operating profit | 22 | 34 | 78
Question: what percent of distribution sales where attributable to printing papers and graphic arts supplies and equipment in 2011?
Rationale: The sales of print papers and graphic arts supplies and equipment in 2011 is 3.5 billion. The total sales in 2011 is 6.63 billion. The percentage is 52.8%. So the answer is 52.8%.
[Final Answer]: 52.8% \n


Read the following text and table, and then answer a question:
Text: Effective Income Tax Rate
A reconciliation of the United States federal statutory income tax rate to our effective income tax rate is as follows:
In 2019 and 2018 we had pre-tax losses of $19,573 and $25,403, respectively, which are available for carry forward to offset future taxable income. We made determinations to provide full valuation allowances for our net deferred tax assets at the end of 2019 and 2018, including NOL carryforwards generated during the years, based on our evaluation of positive and negative evidence, including our history of operating losses and the uncertainty of generating future taxable income that would enable us to realize our deferred tax.
Table: — | Year Ended | Year Ended
— | December 31, 2018 | December 31, 2019
United States federal statutory rate | 21.00% | 21.00%
State taxes, net of federal benefit | 1.99% | -0.01%
Valuation allowance | -21.96% | -24.33%
Cumulative effect of accounting change | — | 2.07%
R&D Credit | 1.34% | 1.53%
Other | -0.38% | -0.27%
Effective income tax rate | 1.99% | -0.01%
Question: What was the 2019 percentage change in pre-tax losses?
Rationale: The pre-tax losses in 2019 is $19,573 and the pre-tax losses in 2018 is $25,403. The net change in pre-tax losses is -$5,830. The percentage change is -22.95%. So the answer is:
[Final Answer]: -22.95% \n

Read the following text and table, and then answer a question
Table: - | september 24 2005 | september 25 2004 | september 27 2003
beginning allowance balance | $ 47 | $ 49 | $ 51
charged to costs and expenses | 8 | 3 | 4
deductions ( a ) | -9 ( 9 ) | -5 ( 5 ) | -6 ( 6 )
ending allowance balance | $ 46 | $ 47 | $ 49
Question: what was the highest ending allowance balance , in millions?
Rationale: The ending allowance balance in 2005 is 47. The ending allowance balance in 2004 is 49. The ending allowance balance in 2003 is 51. The highest ending allowance balance is 51. So the answer is 51.
[Final Answer]: 51 \n
Follow the given examples
Think step by step, use information from given table and text. For answer """+"Table and Text:"+top_3 +"give only numerical or (yes or no) type answers when applicable without narrative for Question: answer the Question:""" +row.question.text() +"In [Final Answer]: give only exact answer without detailed text"
                        #print("user_prompt",user_prompt)

                        answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                if len(answer.split("[Final Answer]:")) >1:
                        if "yes" in answer.split("[Final Answer]:")[-1].lower().split():
                                ans = "yes"
                        elif "no" in answer.split("[Final Answer]:")[-1].lower().split():
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
                        if str(row.answer.text()).lower() == ans:
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
                final_questions.to_csv("chatgpt_finqa_rag_few_shot_cot_5.tsv",sep="\t",index=False)


