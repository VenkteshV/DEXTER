from torch.utils.data import Dataset


class QaDataset(Dataset):
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
    def __init__(self, query_ids, query_mask, context_ids, context_mask):
        """_summary_

        Args:
            query_ids (_type_): _description_
            query_mask (_type_): _description_
            context_ids (_type_): _description_
            context_mask (_type_): _description_
        """
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
