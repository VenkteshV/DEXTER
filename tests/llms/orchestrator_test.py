import unittest

from methods.llms.llm_engine_orchestrator import LLMEngineOrchestrator

from methods.llms.flant5_engine import FlanT5Engine
from methods.llms.openai_engine import OpenAIEngine
from methods.llms.llama_engine import LlamaEngine

class MyTestCase(unittest.TestCase):
    def test_loader(self):
        config_instance = LLMEngineOrchestrator()
        llm_instance = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo")
        self.assertTrue(isinstance(llm_instance, OpenAIEngine))

        llm_instance = config_instance.get_llm_engine(data="",llm_class="flant5",model_name="google/flan-t5-xl")
        self.assertTrue(isinstance(llm_instance, FlanT5Engine))
        answer = llm_instance.get_flant5_completion("generate a song")
        print(answer)
        llm_instance = config_instance.get_llm_engine(data="",llm_class="llama",model_name="meta-llama/Llama-2-7b-hf")
        self.assertTrue(isinstance(llm_instance, LlamaEngine))
        answer = llm_instance.get_llama_completion("Hi! I like cooking. Can you suggest some recipes?")

if __name__ == '__main__':
    unittest.main()
