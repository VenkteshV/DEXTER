from collections import defaultdict
import configparser
import json

import tqdm
from constants import Split
from data.datastructures.dataset import ReaderDataset
from data.loaders.BaseDataLoader import PassageDataLoader
from data.loaders.DataLoaderFactory import DataLoaderFactory
from data.loaders.Tokenizer import Tokenizer
from torch.utils.data import RandomSampler, SequentialSampler


class ReaderDataLoader(DataLoaderFactory):
    def __init__(self, dataset:str,passage_dataset:str,config_path,split:Split, batch_size=32,tokenizer="bert-base-uncased"):
        self.split = split
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.tokenizer_name = tokenizer
        self.tokenizer = Tokenizer(self.tokenizer_name)
        base_dataset = DataLoaderFactory().create_dataloader(dataset, config_path, self.split, batch_size,self.tokenizer_name)
        self.base_dataset = base_dataset
        retreival_out = f'{dataset}-{split}-out'.lower()
        retreival_results_path = self.config['Retrieval'][retreival_out]
        with open(retreival_results_path,"r") as fp:
            self.retreival_results = json.load(fp)[:11]
        psg_ids = [p_idx for retrieved in self.retreival_results for p_idx in retrieved]
        self.passage_dataloader = PassageDataLoader(passage_dataset,psg_ids,self.tokenizer_name,config_path)
        self.tokenized_data = self.load_tokenized_data()
        self.dataset = ReaderDataset(self.tokenized_data)

        if split==Split.TRAIN:
            sampler = RandomSampler(self.dataset)
        else:
            sampler = SequentialSampler(self.dataset)
        print("Dataset loaded of length", len(self.dataset))
        super(ReaderDataLoader, self).__init__(
            self.dataset, sampler=sampler, batch_size=batch_size
        )


    def load_tokenized_data(self,max_n_answers:int=30):
        features = defaultdict(list)
        for i, (q_input_ids, q_attention_mask, retrieved) in \
                tqdm(enumerate(zip(self.base_dataset.dataset.enc_ids, self.base_dataset.dataset.enc_mask, self.retreival_results))):
            assert len(q_input_ids)==len(q_attention_mask)==32
            q_input_ids = [in_ for in_, mask in zip(q_input_ids, q_attention_mask) if mask]
            assert 3<=len(q_input_ids)<=32
            p_input_ids = []
            p_attention_mask = []
            for p_idx in retrieved:
                input_id,att_mask= self.passage_dataloader.dataset.get_by_id(p_idx)
                p_input_ids.append(input_id)
                p_attention_mask.append(att_mask)
            a_input_ids = [ans_ids[1:-1] for ans_ids in self.base_dataset.dataset.dec_ids[i]]
            detected_spans = []
            for _p_input_ids in p_input_ids:
                detected_spans.append([])
                for _a_input_ids in a_input_ids:
                    decoded_a_input_ids = self.base_dataset.tokenizer.decode(_a_input_ids)
                    for j in range(len(_p_input_ids)-len(_a_input_ids)+1):
                        if _p_input_ids[j:j+len(_a_input_ids)]==_a_input_ids:
                            detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+len(_a_input_ids)-1))
                        elif "albert" in "a" and \
                                _p_input_ids[j]==_a_input_ids[0] and \
                                13 in _p_input_ids[j:j+len(_a_input_ids)]:
                            k = j + len(_a_input_ids)+1
                            while k<len(_p_input_ids) and np.sum([_p_input_ids[z]!=13 for z in range(j, k)])<len(_a_input_ids):
                                k += 1
                            if decoded_a_input_ids==self.tokenizer.decode(_p_input_ids[j:k]):
                                detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+k-1))
            if self.split==Split.TRAIN:
                positives = [j for j, spans in enumerate(detected_spans) if len(spans)>0][:20]
                negatives = [j for j, spans in enumerate(detected_spans) if len(spans)==0][:50]
                if len(positives)==0:
                    continue
            else:
                positives = [j for j in range(len(detected_spans))]
                negatives = []
            for key in ["positive_input_ids", "positive_input_mask", "positive_token_type_ids",
                            "positive_start_positions", "positive_end_positions", "positive_answer_mask",
                            "negative_input_ids", "negative_input_mask", "negative_token_type_ids"]:
                features[key].append([])

            def _form_input(p_input_ids, p_attention_mask):
                assert len(p_input_ids)==len(p_attention_mask)
                assert len(p_input_ids)==128 or (len(p_input_ids)<=128 and np.sum(p_attention_mask)==len(p_attention_mask))
                if len(p_input_ids)<128:
                    p_input_ids += [self.tokenizer.tokenizer.pad_token_id for _ in range(128-len(p_input_ids))]
                    p_attention_mask += [0 for _ in range(128-len(p_attention_mask))]
                input_ids = q_input_ids + p_input_ids + [self.tokenizer.tokenizer.pad_token_id for _ in range(32-len(q_input_ids))]
                attention_mask = [1 for _ in range(len(q_input_ids))]  + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
                token_type_ids = [0 for _ in range(len(q_input_ids))] + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
                return input_ids, attention_mask, token_type_ids

            for idx in positives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["positive_input_ids"][-1].append(input_ids)
                features["positive_input_mask"][-1].append(attention_mask)
                features["positive_token_type_ids"][-1].append(token_type_ids)
                detected_span = detected_spans[idx]
                features["positive_start_positions"][-1].append(
                    [s[0] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
                features["positive_end_positions"][-1].append(
                    [s[1] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
                features["positive_answer_mask"][-1].append(
                    [1 for _ in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
            for idx in negatives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["negative_input_ids"][-1].append(input_ids)
                features["negative_input_mask"][-1].append(attention_mask)
                features["negative_token_type_ids"][-1].append(token_type_ids)
        return features