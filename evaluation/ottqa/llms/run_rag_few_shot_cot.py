from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset


if __name__=="__main__":
        config_instance = LLMEngineOrchestrator()
        llm_instance = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo")
        #assertTrue(isinstance(llm_instance, OpenAIEngine))
        with open("/home/venky/venky_bcqa/BCQA/ottqa_colbert_docs.json") as f:
                evidence = json.load(f)
        question_df = {"questions":[],"answers":[]}


        loader = RetrieverDataset("ottqa","ottqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)
    
        queries, qrels, corpus = loader.qrels()
        raw_data = loader.base_dataset.raw_data
        system_prompt = "Given the question, table and text, follow the given examples, think step by step decompose the question and use information in table and text, output final answer for the question and give answer in form of  [Final Answer]: \n"
        matches = 0
        mismatches = 0
        ids = []
        evidences = []

        for index,row in enumerate(raw_data):
                if row.question.id() in ids:
                        continue
                else:
                        ids.append(row.question.id())
                top_3 = " ".join(evidence[row.question.id()][0:5])
                print("top_3***********",len(evidence[row.question.id()][0:5]))
                try:
                        user_prompt = """
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
[Final Answer]:no


Read the following text and table, and then answer a question
Text: annual sales of printing papers and graphic arts supplies and equipment totaled $ 3.5 billion in 2012 compared with $ 4.0 billion in 2011 and $ 4.2 billion in 2010 , reflecting declining demand and the exiting of unprofitable businesses .
Table: in millions | 2012 | 2011 | 2010
sales | $ 6040 | $ 6630 | $ 6735
operating profit | 22 | 34 | 78
Question: what percent of distribution sales where attributable to printing papers and graphic arts supplies and equipment in 2011?
Rationale: The sales of print papers and graphic arts supplies and equipment in 2011 is 3.5 billion. The total sales in 2011 is 6.63 billion. The percentage is 52.8%. So the answer is 52.8%.
[Final Answer]:52.8%


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
[Final Answer]: -22.95%

Read the following text and table, and then answer a question
Table: - | september 24 2005 | september 25 2004 | september 27 2003
beginning allowance balance | $ 47 | $ 49 | $ 51
charged to costs and expenses | 8 | 3 | 4
deductions ( a ) | -9 ( 9 ) | -5 ( 5 ) | -6 ( 6 )
ending allowance balance | $ 46 | $ 47 | $ 49
Question: what was the highest ending allowance balance , in millions?
Rationale: The ending allowance balance in 2005 is 47. The ending allowance balance in 2004 is 49. The ending allowance balance in 2003 is 51. The highest ending allowance balance is 51. So the answer is 51.
[Final Answer]:51
                        Following the give examples, Think step by step, use information from given table and text"""+"Table and Text:"+top_3 +"and answer the Question:""" +row.question.text()
                        #print("user_prompt",user_prompt)

                        chain_answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                # if "not possible" in chain_answer.lower():
                #         mismatches+=1
                #         continue
                # elif "unknown" in chain_answer.lower():
                #         mismatches+=1
                #         continue
                except:
                        top_3 = " ".join(evidence[row.question.id()][0:5])

                        user_prompt = """
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
[Final Answer]:no


Read the following text and table, and then answer a question
Text: annual sales of printing papers and graphic arts supplies and equipment totaled $ 3.5 billion in 2012 compared with $ 4.0 billion in 2011 and $ 4.2 billion in 2010 , reflecting declining demand and the exiting of unprofitable businesses .
Table: in millions | 2012 | 2011 | 2010
sales | $ 6040 | $ 6630 | $ 6735
operating profit | 22 | 34 | 78
Question: what percent of distribution sales where attributable to printing papers and graphic arts supplies and equipment in 2011?
Rationale: The sales of print papers and graphic arts supplies and equipment in 2011 is 3.5 billion. The total sales in 2011 is 6.63 billion. The percentage is 52.8%. So the answer is 52.8%.
[Final Answer]:52.8%



Read the following text and table, and then answer a question:
Text: 11 Intangible assets (continued)
(a) Intangible assets
RIGHTS AND LICENCES
Certain licences that NEXTDC possesses have an indefinite useful life and are carried at cost less impairment losses and are subject to impairment review at least annually and whenever there is an indication that it may be impaired.
Other licences that NEXTDC acquires are carried at cost less accumulated amortisation and accumulated impairment losses. Amortisation is recognised on a straight-line basis over the estimated useful life. The estimated useful life and amortisation method are reviewed at the end of each annual reporting period.
INTERNALLY GENERATED SOFTWARE
Internally developed software is capitalised at cost less accumulated amortisation. Amortisation is calculated using the straight-line basis over the asset’s useful economic life which is generally two to three years. Their useful lives and potential impairment are reviewed at the end of each financial year.
SOFTWARE UNDER DEVELOPMENT
Costs incurred in developing products or systems and costs incurred in acquiring software and licenses that will contribute to future period financial benefits through revenue generation and/or cost reduction are capitalised to software and systems. Costs capitalised include external direct costs of materials and services and employee costs.
Assets in the course of construction include only those costs directly attributable to the development phase and are only recognised following completion of technical feasibility and where the Group has an intention and ability to use the asset.
Table: — | Rights and licenses | Internally generated software | Software under development | Total
Movements | $'000 | $'000 | $'000 | $'000
At 30 June 2019 | — | — | — | —
Cost | 13 | 12,961 | 16,284 | 29,259
Accumulated amortisation | - | -5,580 | - | -5,580
Netbook amount | 13 | 7,381 | 16,284 | 23,678
30 June 2018 | — | — | — | —
Opening net book amount at 1 July 2017 | 43 | 442 | 8,053 | 8,538
Additions – externally acquired | 13 | - | 5,253 | 5,266
Additions – internally developed | - | - | 1,256 | 1,256
Amortisation | -43 | -1,746 | - | -1,789
Transfers | - | 7,563 | -7,563 | -
Transfer between classes | - | 744 | - | 744
Disposals | - | -618 | -490 | -1,108
Closing net book amount | 13 | 6,385 | 6,509 | 12,907
At 30 June 2018 | — | — | — | —
Cost | 104 | 9,555 | 6,509 | 16,168
Accumulated amortisation | -91 | -3,170 | - | -3,261
Net book amount | 13 | 6,385 | 6,509 | 12,907
Question: Which year have greater total accumulated amortisation?
Rationale: The total accumulated amortisation in 2019 is $5,580 thousand and the total accumulated amortisation in 2018 is $3,261 thousand. 2019 has greater total accumulated amortisation. So the answer is:
[Final Answer]: 2019


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
[Final Answer]: -22.95%

Read the following text and table, and then answer a question
Table: - | september 24 2005 | september 25 2004 | september 27 2003
beginning allowance balance | $ 47 | $ 49 | $ 51
charged to costs and expenses | 8 | 3 | 4
deductions ( a ) | -9 ( 9 ) | -5 ( 5 ) | -6 ( 6 )
ending allowance balance | $ 46 | $ 47 | $ 49
Question: what was the highest ending allowance balance , in millions?
Rationale: The ending allowance balance in 2005 is 47. The ending allowance balance in 2004 is 49. The ending allowance balance in 2003 is 51. The highest ending allowance balance is 51. So the answer is 51.
[Final Answer]:51
                        Following the give examples, Think step by step, use information from given table and text"""+"Table and Text:"+top_3 +"and answer the Question:""" +row.question.text()
                        #print("user_prompt",user_prompt)

                        chain_answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                if len(chain_answer.split("[Final Answer]:")) >0:
                        answer = chain_answer.split("[Final Answer]:")[-1]
                        print("************",answer,row.answer.text())
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
                final_questions.to_csv("chatgpt_ottqa_rag_few_shot_cot_5.tsv",sep="\t",index=False)


