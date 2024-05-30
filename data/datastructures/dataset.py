from torch.utils.data import Dataset



class QADataset(Dataset):
    """
    QA dataset for tokenized questions and corresponding answers to train answering models
    """
    def __init__(self, enc_ids, enc_mask, dec_ids, dec_mask, is_training):
        assert len(enc_ids) == len(enc_mask)
        assert len(dec_ids) == len(dec_mask)
        self.enc_ids = enc_ids
        self.enc_mask = enc_mask
        self.dec_ids = dec_ids
        self.dec_mask = dec_mask
        self.is_training = is_training

    def __len__(self):
        return len(self.enc_ids)

    def __getitem__(self, idx):
        if self.is_training:
            return self.enc_ids[idx], self.enc_mask[idx], self.dec_ids[idx], self.dec_mask[idx]
        else:
            return self.enc_ids[idx], self.enc_mask[idx]


class DprDataset(Dataset):
    """ Dataset to hold tokenized Question and Related Context

    Args:
        query_ids : input ids of all tokenized questions
        query_mask : attention mask for tokenized question
        context_ids : input ids of tokenized context
        context_mask : attention mask for tokenized context
    """
    def __init__(self, query_ids, query_mask, context_ids, context_mask):
        assert len(query_ids) == len(query_mask)
        assert len(context_ids) == len(context_mask)
        self.query_ids = query_ids
        self.query_mask = query_mask
        self.context_ids = context_ids
        self.context_mask = context_mask

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        return self.query_ids[idx], self.query_mask[idx], self.context_ids[idx], self.context_mask[idx]

class PassageDataset(Dataset):
    """ Dataset to load and process an corpus containing passages that can serve as contexts for question answering.

    Args:
        passage_ids : ids of passages in the corpus for identification
        input_ids : input ids of context post tokenization
        attention_mask : attention mask for context post tokenization
        """
    def __init__(self,passage_ids,input_ids,attention_mask):
        assert len(input_ids)==len(passage_ids)==len(attention_mask)
        self.input_ids = input_ids
        self.passage_ids = passage_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.passage_ids)

    def __getitem__(self, idx):
        assert idx in self.passage_ids
        return self.input_ids[idx], self.attention_mask[idx]

    def get_by_id(self,idx):
        return self.__getitem__(idx)
