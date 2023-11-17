from constants import Dataset
from data.loaders.AmbigQADataLoader import AmbigQADataLoader
from data.loaders.FinQADataLoader import FinQADataLoader
from data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader



class DataLoaderFactory:
    def __int__(self):
        pass

    def create_dataloader(self, dataloader_name,config_path,split,batch_size,tokenizer):
        if Dataset.AMBIGQA in dataloader_name:
            loader = AmbigQADataLoader
        elif(Dataset.FINQA) in dataloader_name:
            loader = FinQADataLoader
        elif(Dataset.WIKIMULTIHOPQA) in dataloader_name:
            loader = WikiMultihopQADataLoader
        
        else:
            raise NotImplemented(f"{dataloader_name} not implemented yet.")
        return loader(dataloader_name,tokenizer, config_path,split,batch_size)