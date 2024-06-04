"""
 LLM orchestrator factory that decides type of LLm engine to use
"""

import logging


from dexter.llms.flant5_engine import FlanT5Engine
from dexter.llms.llama_engine import LlamaEngine
from dexter.llms.mistral_engine import MistralEngine
from dexter.llms.openai_engine import OpenAIEngine

logger = logging.getLogger()
from dexter.config import LLmConstants

class LLMEngineOrchestrator:
    @staticmethod
    def get_llm_engine(llm_class, data,model_name=None, temperature=0.3,top_n = 1, top_p = None):
        if llm_class.lower() == LLmConstants.FLANT5.lower():
            return FlanT5Engine(data,model_name,temperature,top_n)

        elif llm_class.lower() == LLmConstants.OPENAI.lower():
            return OpenAIEngine(data,model_name,temperature,top_n)
        elif llm_class.lower() == LLmConstants.LLAMA.lower():
            return LlamaEngine(data, model_name, temperature, top_n)
        elif llm_class.lower() == LLmConstants.MISTRAL.lower():
            return MistralEngine(data,model_name,temperature,top_n)
        else:
            classname = llm_class
            if classname not in globals():
                raise ValueError("No implementation found for the custom class specified: {}".format(classname))
            return globals()[classname]()
