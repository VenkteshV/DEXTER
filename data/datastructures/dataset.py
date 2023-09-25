from torch.utils.data import Dataset


class MyDataset(Dataset):
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

