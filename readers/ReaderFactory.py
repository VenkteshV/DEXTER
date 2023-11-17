from collections import defaultdict
from enum import Enum
import math
from typing import List
from joblib import Parallel, delayed
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, AlbertConfig, BartConfig
from data.datastructures.dataset import ReaderDataset
from data.loaders.ReaderDataLoader import ReaderDataLoader



from metrics.MetricsBase import Metric
from readers.Seq2SeqPredictiors import Bart
from readers.SpanPredictors import AlbertSpanPredictor, BertSpanPredictor



class ReaderName(Enum):
    BERT = "bert"
    ALBERT = "albert"
    BART = "bart"


class Reader:
    def __init__(self, model):
        self.model = model

    def train(self):
        pass

    def infer(self,data_loader:ReaderDataLoader,n_paragraphs:int=None):
        self.model.eval()
        outputs = []
        data_loader_tq = tqdm(data_loader)
        for i, batch in enumerate(data_loader_tq):
            with torch.no_grad():
                batch = [b.to(torch.device("cpu")) for b in batch]
                batch_start_logits, batch_end_logits, batch_sel_logits = self.model(
                    input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])
                batch_start_logits = batch_start_logits.detach().cpu().tolist()
                batch_end_logits = batch_end_logits.detach().cpu().tolist()
                batch_sel_logits = batch_sel_logits.detach().cpu().tolist()
                assert len(batch_start_logits)==len(batch_end_logits)==len(batch_sel_logits)
                for start_logit, end_logit, sel_logit in zip(batch_start_logits, batch_end_logits, batch_sel_logits):
                    outputs.append((start_logit, end_logit, sel_logit))
        predictions = self.decode(outputs,data_loader.tokenizer.tokenizer,data_loader.tokenized_data,
                                       n_paragraphs=n_paragraphs)
        return predictions


    def evaluate(self,data_loader:ReaderDataLoader,metrics:List[Metric],n_paragraphs:int=None):
        predictions = self.infer(data_loader,n_paragraphs)
        results = {}
        for metric in metrics:
            if n_paragraphs is None:
                ems = []
                for prediction, dp in zip(predictions, data_loader.base_dataset.raw_data):
                    if type(prediction) == list:
                        prediction = prediction[0]
                    if type(prediction) == dict:
                        prediction = prediction["text"]
                    ems.append(metric.evaluate(prediction, dp.answer.flatten()))
            else:
                ems = defaultdict(list)
                for prediction, dp in zip(predictions, data_loader.base_dataset.raw_data):
                    assert len(n_paragraphs) == len(prediction)
                    for pred, n in zip(prediction, n_paragraphs):
                        if type(pred) == list:
                            pred = pred[0]
                        if type(pred) == dict:
                            pred = pred["text"]
                        ems[n].append(metric.evaluate(pred, dp["answer"].flatten()))
                for n in n_paragraphs:
                    print("n_paragraphs=%d\t#M=%.2f" % (n, np.mean(ems[n]) * 100))
                ems = ems[n_paragraphs[-1]]
            results[metric.name()]= np.mean(ems)
        return results

    def decode_span_batch(self,features, scores, tokenizer, max_answer_length,
                      n_paragraphs=None, topk_answer=1, verbose=False, n_jobs=1,
                      save_psg_sel_only=False):
        assert len(features)==len(scores)
        iter=zip(features, scores)
        if n_jobs>1:
            def f(t):
                return self.decode_span(t[0], tokenizer, t[1][0], t[1][1], t[1][2], max_answer_length,
                                n_paragraphs=n_paragraphs, topk_answer=topk_answer,
                                save_psg_sel_only=save_psg_sel_only)
            return Parallel(n_jobs=n_jobs)(delayed(f)(t) for t in iter)
        if verbose:
            iter = tqdm(iter)
        predictions = [self.decode(feature, tokenizer, start_logits, end_logits, sel_logits,
                            max_answer_length, n_paragraphs, topk_answer, save_psg_sel_only) \
                for (feature, (start_logits, end_logits, sel_logits)) in iter]
        return predictions

    def decode_span(self,feature, tokenizer, start_logits_list, end_logits_list, sel_logits_list,
                max_answer_length, n_paragraphs=None, topk_answer=1, save_psg_sel_only=False):
        all_positive_token_ids, all_positive_input_mask = feature
        assert len(start_logits_list)==len(end_logits_list)==len(sel_logits_list)
        assert type(sel_logits_list[0])==float
        log_softmax_switch_logits_list = self._compute_log_softmax(sel_logits_list[:len(all_positive_token_ids)])

        if save_psg_sel_only:
            return np.argsort(-np.array(log_softmax_switch_logits_list)).tolist()

        sorted_logits = sorted(enumerate(zip(start_logits_list, end_logits_list, sel_logits_list)),
                            key=lambda x: -x[1][2])
        nbest = []
        for passage_index, (start_logits, end_logits, switch_logits) in sorted_logits:
            scores = []
            if len(all_positive_token_ids)<=passage_index:
                continue

            positive_token_ids = all_positive_token_ids[passage_index]
            positive_input_mask = all_positive_input_mask[passage_index]
            start_offset = 1 + positive_token_ids.index(tokenizer.sep_token_id)
            end_offset = positive_input_mask.index(0) if 0 in positive_input_mask else len(positive_input_mask)

            positive_token_ids = positive_token_ids[start_offset:end_offset]
            start_logits = start_logits[start_offset:end_offset]
            end_logits = end_logits[start_offset:end_offset]
            log_softmax_start_logits = self._compute_log_softmax(start_logits)
            log_softmax_end_logits = self._compute_log_softmax(end_logits)

            for (i, s) in enumerate(start_logits):
                for (j, e) in enumerate(end_logits[i:i+max_answer_length]):
                    scores.append(((i, i+j), s+e))

            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            chosen_span_intervals = []

            for (start_index, end_index), score in scores:
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                if any([start_index<=prev_start_index<=prev_end_index<=end_index or
                        prev_start_index<=start_index<=end_index<=prev_end_index
                        for (prev_start_index, prev_end_index) in chosen_span_intervals]):
                    continue

                answer_text = tokenizer.decode(positive_token_ids[start_index:end_index+1],
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True).strip()
                passage_text = tokenizer.decode(positive_token_ids[:start_index],
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True).strip() + \
                    " <answer>" + answer_text + "</answer> " + \
                    tokenizer.decode(positive_token_ids[end_index+1:],
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True).strip()

                nbest.append({
                    'text': answer_text,
                    'passage_index': passage_index,
                    'passage': passage_text,
                    'log_softmax': log_softmax_switch_logits_list[passage_index] + \
                                    log_softmax_start_logits[start_index] + \
                                    log_softmax_end_logits[end_index]})

                chosen_span_intervals.append((start_index, end_index))
                if topk_answer>-1 and topk_answer==len(chosen_span_intervals):
                    break

        if len(nbest)==0:
            nbest = [{'text': 'empty', 'log_softmax': -99999, 'passage_index': 0, 'passage': ''}]

        sorted_nbest = sorted(nbest, key=lambda x: -x["log_softmax"])

        if n_paragraphs is None:
            return sorted_nbest[:topk_answer] if topk_answer>-1 else sorted_nbest
        else:
            return [[pred for pred in sorted_nbest if pred['passage_index']<n] for n in n_paragraphs]

    def decode(self,outputs,tokenizer,tokenized_data, n_paragraphs=50,max_ans_length=10,topk_answer=1,verbose=False,n_jobs=12):
        return self.decode_span_batch(list(zip(tokenized_data["positive_input_ids"],
                                        tokenized_data["positive_input_mask"])),
                                outputs,
                                tokenizer=tokenizer,
                                max_answer_length=max_ans_length,
                                n_paragraphs=n_paragraphs,
                                topk_answer=topk_answer,
                                verbose=verbose,
                                n_jobs=n_jobs,
                                save_psg_sel_only=False)

    def _compute_log_softmax(self,scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []
        if type(scores[0])==tuple:
            scores = [s[1] for s in scores]
        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score
        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x
        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return np.log(probs).tolist()



class ReaderFactory:
    def __int__(self):
        pass

    def create_reader(self, reader_name, base,resize_embeddings:int=None, checkpoint=None):
        if reader_name == ReaderName.BERT.name:
            Config = BertConfig
            Model = BertSpanPredictor
        elif reader_name == ReaderName.ALBERT.name:
            Config = AlbertConfig
            Model = AlbertSpanPredictor
        elif reader_name == ReaderName.BART.name:
            Config = BartConfig
            Model = Bart
        else:
            raise NotImplemented(f"{reader_name} not implemented yet.")

        if checkpoint:
            model = self._load_from_checkpoint(checkpoint,reader_name, base, Model, Config,resize_embeddings)
        else:
            model = Model.from_pretrained(base)
        return Reader(model)

    def _load_from_checkpoint(self, checkpoint,reader_name, base, Model, Config,resize_embeddings):
        # TODO: DeepaliP98GPU handling
        def convert_to_single_gpu(state_dict):
            if "model_dict" in state_dict:
                state_dict = state_dict["model_dict"]
            def _convert(key):
                if key.startswith("module."):
                    return key[7:]
                return key

            return {_convert(key): value for key, value in state_dict.items()}

        state_dict = convert_to_single_gpu(torch.load(checkpoint))
        model = Model(Config.from_pretrained(base))
        if ReaderName.BART.name in reader_name:
            model.resize_token_embeddings(resize_embeddings)
        print("Loading from {}".format(checkpoint))
        return model.from_pretrained(None, config=model.config, state_dict=state_dict)
