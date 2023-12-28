from constants import Dataset, Split
from data.loaders.AmbigQADataLoader import AmbigQADataLoader
from data.loaders.FinQADataLoader import FinQADataLoader
from data.loaders.OTTQADataLoader import OTTQADataLoader
from data.loaders.TATQADataLoader import TATQADataLoader
from data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader



class DataLoaderFactory:
    def __int__(self):
        pass

    def create_dataloader(
        self,
        dataloader_name: str,
        tokenizer="bert-base-uncased",
        config_path="test_config.ini",
        split=Split.TRAIN,
        batch_size=None,
    ):
        if Dataset.AMBIGQA in dataloader_name:
            loader = AmbigQADataLoader
        elif Dataset.FINQA in dataloader_name:
            loader = FinQADataLoader
        elif Dataset.WIKIMULTIHOPQA in dataloader_name:
            loader = WikiMultihopQADataLoader
        elif Dataset.TATQA in dataloader_name:
            loader = TATQADataLoader  
        elif Dataset.OTTQA in dataloader_name:
            loader = OTTQADataLoader
        else:
            raise NotImplemented(f"{dataloader_name} not implemented yet.")
        return loader(dataloader_name,tokenizer, config_path,split,batch_size)