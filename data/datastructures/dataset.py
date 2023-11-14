from torch.utils.data import Dataset


class QADataset(Dataset):
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

class PassageDataset(Dataset):
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

class ReaderDataset(Dataset):
    def __init__(self, data,
                 is_training=False, train_M=16, test_M=16):
        self.data = data
        self.positive_input_ids = self.tensorize("positive_input_ids")
        self.positive_input_mask = self.tensorize("positive_input_mask")
        self.positive_token_type_ids = self.tensorize("positive_token_type_ids")
        assert len(self.positive_input_ids)==len(self.positive_input_mask)==len(self.positive_token_type_ids)

        if is_training:
            self.positive_start_positions = self.tensorize("positive_start_positions")
            self.positive_end_positions = self.tensorize("positive_end_positions")
            self.positive_answer_mask = self.tensorize("positive_answer_mask")
            self.negative_input_ids = self.tensorize("negative_input_ids")
            self.negative_input_mask = self.tensorize("negative_input_mask")
            self.negative_token_type_ids = self.tensorize("negative_token_type_ids")
            assert len(self.negative_input_ids)==len(self.negative_input_mask)==len(self.negative_token_type_ids)
            assert len(self.positive_input_ids)==\
                    len(self.positive_start_positions)==len(self.positive_end_positions)==len(self.positive_answer_mask)
            assert all([len(positive_input_ids)>0 for positive_input_ids in self.positive_input_ids])

        self.is_training = is_training
        self.train_M = train_M
        self.test_M = test_M

    def __len__(self):
        return len(self.positive_input_ids)

    def __getitem__(self, idx):
        if not self.is_training:
            input_ids = self.positive_input_ids[idx][:self.test_M]
            input_mask = self.positive_input_mask[idx][:self.test_M]
            token_type_ids = self.positive_token_type_ids[idx][:self.test_M]
            return [self._pad(t, self.test_M) for t in [input_ids, input_mask, token_type_ids]]

        # sample positive
        positive_idx = np.random.choice(len(self.positive_input_ids[idx]))
        #positive_idx = 0
        positive_input_ids = self.positive_input_ids[idx][positive_idx]
        positive_input_mask = self.positive_input_mask[idx][positive_idx]
        positive_token_type_ids = self.positive_token_type_ids[idx][positive_idx]
        positive_start_positions = self.positive_start_positions[idx][positive_idx]
        positive_end_positions = self.positive_end_positions[idx][positive_idx]
        positive_answer_mask = self.positive_answer_mask[idx][positive_idx]

        # sample negatives
        negative_idxs = np.random.permutation(range(len(self.negative_input_ids[idx])))[:self.train_M-1]
        negative_input_ids = [self.negative_input_ids[idx][i] for i in negative_idxs]
        negative_input_mask = [self.negative_input_mask[idx][i] for i in negative_idxs]
        negative_token_type_ids = [self.negative_token_type_ids[idx][i] for i in negative_idxs]
        negative_input_ids, negative_input_mask, negative_token_type_ids = \
            [self._pad(t, self.train_M-1) for t in [negative_input_ids, negative_input_mask, negative_token_type_ids]]

        # aggregate
        input_ids = torch.cat([positive_input_ids.unsqueeze(0), negative_input_ids], dim=0)
        input_mask = torch.cat([positive_input_mask.unsqueeze(0), negative_input_mask], dim=0)
        token_type_ids = torch.cat([positive_token_type_ids.unsqueeze(0), negative_token_type_ids], dim=0)
        start_positions, end_positions, answer_mask = \
            [self._pad([t], self.train_M) for t in [positive_start_positions,
                                                  positive_end_positions,
                                                  positive_answer_mask]]
        return input_ids, input_mask, token_type_ids, start_positions, end_positions, answer_mask

    def tensorize(self, key):
        return [torch.LongTensor(t) for t in self.data[key]] if key in self.data.keys() else None

    def _pad(self, input_ids, M):
        if len(input_ids)==0:
            return torch.zeros((M, self.negative_input_ids[0].size(1)), dtype=torch.long)
        if type(input_ids)==list:
            input_ids = torch.stack(input_ids)
        if len(input_ids)==M:
            return input_ids
        return torch.cat([input_ids,
                          torch.zeros((M-input_ids.size(0), input_ids.size(1)), dtype=torch.long)],
                         dim=0)

