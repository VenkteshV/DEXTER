from constants import Dataset
from data.loaders import AmbigQADataLoader


class DataLoaderFactory:
    def __int__(self):
        pass

    def create_dataloader(self, dataloader_name,config_path,split,batch_size,tokenizer):
        if Dataset.AMBIGQA in dataloader_name:
            loader = AmbigQADataLoader
        else:
            raise NotImplemented(f"{dataloader_name} not implemented yet.")
        return loader(dataloader_name,tokenizer, config_path,split,batch_size)