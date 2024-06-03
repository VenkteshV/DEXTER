'''Base class for a retriever'''
class BaseRetriver:
    def __init__(self) -> None:
        pass

    def encode_queries(self)->None:
        pass

    def encode_context(self)->None:
        pass

    def train(self)->None:
        pass

    def retrieve(self)->None:
        pass

'''Retriever Factory to get retriever class given alias name'''

class RetrieverFactory():

    def get_retreiver(self,retriever_name:str,config_path:str):
        return None

