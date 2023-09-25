import openai

import pandas as pd

openai.api_key = "" #os.getenv("OPENAI_API_KEY")



def get_expansion_llm(query,text):

    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"You are an expert system that expands a query with multiple intents to correct oen based on context \n"},
                      {"role":"user","content": "Expand the given query with short natural language description:"+query+" considering the given context:"+text+"Give expansion:" } 
    ] ,
            temperature=0.3,
            max_tokens=35,
            top_p=1.0,
            frequency_penalty=0.8,
            presence_penalty=0.6
            )

    return response['choices'][0]['message']['content']