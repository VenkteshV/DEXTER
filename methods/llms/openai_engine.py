import openai


openai.api_key = ""

class OpenAIEngine:
    def __init__(self, data,model_name: str, temperature: int, top_n: int):
        self.model_name = model_name
        self.temperature = temperature
        self.top_n = top_n
        self.data = data
    def get_completion(self, prompt: str) -> str:
        """_summary_

        Args:
            prompt (_type_): instruction with possibly in-context samples
        """        
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="""You are a assistant that gives an answer along with the derivation of rationale in format Rationale: Answer:  to arrive at solution for questions mandatorily using information from both given table and text. In table columns are separated by | and rows by \n (newline). If you dont know the answer output UNKNOWN: \n
                      give best answer with accurate scale and precision for Question: """ + row["question"]+",Table: "+row["table"]+ "Text: "+row["text"]+"Output in format Rationale:, Answer:",
            temperature=self.temperature,
            max_tokens=2048,
            top_p=1.0,
            n = self.top_n,
            frequency_penalty=0.8,
            presence_penalty=0.6
            ) 
        return response['choices'][0]['text']     
    def get_chat_completion(self, user_prompt: str, system_prompt: str) -> str:
        """invokes chat completion endpoint to get chatpgt results

        Args:
            user_prompt (str): The user prompt with possibly in context samples
            system_prompt (str): system prompt that designates the responsibility of the system
        """        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":system_prompt } ,
    {"role":"user","content": user_prompt,
    }] ,
            temperature=self.temperature,
            max_tokens=1000,
            top_p=1.0,
            n=self.top_n,
            frequency_penalty=0.8,
            presence_penalty=0.6,
            )
        return response['choices'][0]['text']