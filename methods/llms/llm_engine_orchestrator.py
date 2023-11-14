"""
 LLM orchestrator factory that decides type of LLm engine to use
"""

import os
import time
import logging
import pickle
from typing import List, Tuple, Iterator

import faiss
import numpy as np

logger = logging.getLogger()


from data.datastructures import LLmConstants
import time
import numpy as np
from methods.llms.openai_engine import OpenAIEngine
from methods.llms.llama_engine import LlamaEngine
from methods.llms.flant5_engine import FlanT5Engine

class LLMEngineOrchestrator:
    @staticmethod
    def get_llm_engine(llm_class, data,model_name=None, temperature=0.3,top_n = 1, top_p = None):
        if llm_class.lower() == LLmConstants.FLANT5.lower():
            return FlanT5Engine(data,model_name,temperature,top_n)

        elif llm_class.lower() == LLmConstants.OPENAI.lower():
            return OpenAIEngine(data,model_name,temperature,top_n)
        elif llm_class.lower() == LLmConstants.LLAMA.lower():
            return LlamaEngine(data,model_name,temperature,top_n)
        else:
            classname = llm_class
            if classname not in globals():
                raise ValueError("No implementation found for the custom class specified: {}".format(classname))
            return globals()[classname]()

  