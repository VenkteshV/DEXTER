from dexter.config.constants import Dataset, Split
from dexter.data.loaders.AmbigQADataLoader import AmbigQADataLoader
from dexter.data.loaders.FinQADataLoader import FinQADataLoader
from dexter.data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader

from dexter.data.loaders.OTTQADataLoader import OTTQADataLoader
from dexter.data.loaders.TATQADataLoader import TATQADataLoader
from dexter.data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader
from dexter.data.loaders.StrategyQADataLoader import StrategyQADataLoader



class DataLoaderFactory:
    ''' Data Loader factory to map dataset alias to corresponding Data loader class'''


    def create_dataloader(
        self,
        dataloader_name: str,
        tokenizer="bert-base-uncased",
        config_path="test_config.ini",
        split=Split.TRAIN,
        batch_size=None,
        corpus=None
    ):
        if Dataset.AMBIGQA in dataloader_name:
            loader = AmbigQADataLoader
        elif Dataset.FINQA in dataloader_name:
            loader = FinQADataLoader
        elif Dataset.WIKIMULTIHOPQA in dataloader_name:
            loader = WikiMultihopQADataLoader
        elif Dataset.MUSIQUEQA in dataloader_name:
            loader = MusiqueQADataLoader
        elif Dataset.TATQA in dataloader_name:
            loader = TATQADataLoader  
        elif Dataset.OTTQA in dataloader_name:
            loader = OTTQADataLoader
        elif Dataset.StrategyQA in dataloader_name:
            loader = StrategyQADataLoader
        else:
            raise NotImplemented(f"{dataloader_name} not implemented yet.")
        return loader(dataset=dataloader_name, config_path=config_path,
                      split=split,batch_size=batch_size,
                      tokenizer=tokenizer,
                      corpus=corpus)