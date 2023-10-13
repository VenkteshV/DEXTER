import gzip
import json
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from constants import Split
from data.datastructures.dataset import AmbigQAReaderDataset
from data.loaders.BasedataLoader import AmbigQADataLoader, ReaderDataLoader
from data.readers.ReaderFactory import ReaderFactory, ReaderName

# def load_db(data_path, subset=None):
#         passages = {}
#         titles = {}
#         with gzip.open(data_path, "rb") as f:
#             _ = f.readline()
#             offset = 0
#             for line in tqdm(f):
#                 if subset is None or offset in subset:
#                     _id, passage, title = line.decode().strip().split("\t")
#                     assert int(_id) - 1 == offset
#                     passages[offset] = passage.lower()
#                     titles[offset] = title.lower()
#                 offset += 1
#         assert subset is None or len(subset) == len(titles) == len(passages)
#         print("Loaded {} passages".format(len(passages)))
#         return passages,titles
#
# loader = AmbigQADataLoader("ambignq-light", config_path="data/loaders/test_config.ini", split=Split.DEV, batch_size=10)
#
#
#
# ret_results = json.load(open("C:/Users/deepa/Desktop/MSc CS/Research Assistantship/Project/AmbigQA/AmbigQA/codes/out/dpr/dev_20200201_predictions_1.json"))
# print(len(ret_results))
#
# ret_results = ret_results[:11]
#
# ids = set(p_idx for retrieved in ret_results for p_idx in retrieved)
#
# data_path = "C:/Users/deepa/Desktop/MSc CS/Research Assistantship/Project/AmbigQA/AmbigQA/codes/dpr_data_dir/data/wikipedia_split/psgs_w100_20200201.tsv.gz"
#
# k = json.load(open('data/wikidata.json'))
# passages,titles = k['passages'],k['titles']
#
# psg_ids = list(ids)
# input_data = [titles[str(_id)] + " " + "[SEP]" + " " + passages[str(_id)]
#                           for _id in psg_ids]
# tokenized_data = loader.tokenizer.tokenize(input_data,
#                         max_length=128,
#                         pad_to_max_length=True)
# input_ids = {_id: _input_ids
#                          for _id, _input_ids in zip(psg_ids, tokenized_data["input_ids"])}
# attention_mask = {_id: _attention_mask
#                               for _id, _attention_mask in zip(psg_ids, tokenized_data["attention_mask"])}
# final_tokenized_data = {"input_ids": input_ids, "attention_mask": attention_mask}
#
# features = defaultdict(list)
# max_n_answers = 30
# oracle_exact_matches = []
# flatten_exact_matches = []
# positive_contains_gold_title = []
# for i, (q_input_ids, q_attention_mask, retrieved) in \
#         tqdm(enumerate(zip(loader.dataset.enc_ids, loader.dataset.enc_mask, ret_results))):
#     assert len(q_input_ids)==len(q_attention_mask)==32
#     q_input_ids = [in_ for in_, mask in zip(q_input_ids, q_attention_mask) if mask]
#     assert 3<=len(q_input_ids)<=32
#     p_input_ids = [final_tokenized_data["input_ids"][p_idx] for p_idx in retrieved]
#     p_attention_mask = [final_tokenized_data["attention_mask"][p_idx] for p_idx in retrieved]
#     a_input_ids = [ans_ids[1:-1] for ans_ids in loader.dataset.dec_ids[i]]
#     detected_spans = []
#     for _p_input_ids in p_input_ids:
#         detected_spans.append([])
#         for _a_input_ids in a_input_ids:
#             decoded_a_input_ids = loader.tokenizer.decode(_a_input_ids)
#             for j in range(len(_p_input_ids)-len(_a_input_ids)+1):
#                 if _p_input_ids[j:j+len(_a_input_ids)]==_a_input_ids:
#                     detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+len(_a_input_ids)-1))
#                 elif "albert" in "a" and \
#                         _p_input_ids[j]==_a_input_ids[0] and \
#                         13 in _p_input_ids[j:j+len(_a_input_ids)]:
#                     k = j + len(_a_input_ids)+1
#                     while k<len(_p_input_ids) and np.sum([_p_input_ids[z]!=13 for z in range(j, k)])<len(_a_input_ids):
#                         k += 1
#                     if decoded_a_input_ids==loader.tokenizer.decode(_p_input_ids[j:k]):
#                         detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+k-1))
#     # if self.args.ambigqa and self.is_training:
#     #     positives = [j for j, spans in enumerate(detected_spans) if len(spans)>0][:20]
#     #     negatives = [j for j, spans in enumerate(detected_spans) if len(spans)==0][:50]
#     #     if len(positives)==0:
#     #         continue
#     # elif self.is_training:
#     #     gold_title = normalize_answer(gold_titles[i])
#     #     _positives = [j for j, spans in enumerate(detected_spans) if len(spans)>0]
#     #     if len(_positives)==0:
#     #         continue
#     #     positives = [j for j in _positives if normalize_answer(self.decode(p_input_ids[j][:p_input_ids[j].index(eos_token_id)]))==gold_title]
#     #     positive_contains_gold_title.append(len(positives)>0)
#     #     if len(positives)==0:
#     #         positives = _positives[:20]
#     #     negatives = [j for j, spans in enumerate(detected_spans) if len(spans)==0][:50]
#     # else:
#     positives = [j for j in range(len(detected_spans))]
#     negatives = []
#     for key in ["positive_input_ids", "positive_input_mask", "positive_token_type_ids",
#                 "positive_start_positions", "positive_end_positions", "positive_answer_mask",
#                 "negative_input_ids", "negative_input_mask", "negative_token_type_ids"]:
#         features[key].append([])
#
#     def _form_input(p_input_ids, p_attention_mask):
#         assert len(p_input_ids)==len(p_attention_mask)
#         assert len(p_input_ids)==128 or (len(p_input_ids)<=128 and np.sum(p_attention_mask)==len(p_attention_mask))
#         if len(p_input_ids)<128:
#             p_input_ids += [loader.tokenizer.tokenizer.pad_token_id for _ in range(128-len(p_input_ids))]
#             p_attention_mask += [0 for _ in range(128-len(p_attention_mask))]
#         input_ids = q_input_ids + p_input_ids + [loader.tokenizer.tokenizer.pad_token_id for _ in range(32-len(q_input_ids))]
#         attention_mask = [1 for _ in range(len(q_input_ids))]  + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
#         token_type_ids = [0 for _ in range(len(q_input_ids))] + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
#         return input_ids, attention_mask, token_type_ids
#
#     for idx in positives:
#         input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
#         features["positive_input_ids"][-1].append(input_ids)
#         features["positive_input_mask"][-1].append(attention_mask)
#         features["positive_token_type_ids"][-1].append(token_type_ids)
#         detected_span = detected_spans[idx]
#         features["positive_start_positions"][-1].append(
#             [s[0] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
#         features["positive_end_positions"][-1].append(
#             [s[1] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
#         features["positive_answer_mask"][-1].append(
#             [1 for _ in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
#     for idx in negatives:
#         input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
#         features["negative_input_ids"][-1].append(input_ids)
#         features["negative_input_mask"][-1].append(attention_mask)
#         features["negative_token_type_ids"][-1].append(token_type_ids)
# tokenized_data = features
def inference_span_predictor(model, dev_data, save_predictions=False):
    outputs = []
    if True:
        dataloader = tqdm(dev_data)
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cpu")) for b in batch]
            batch_start_logits, batch_end_logits, batch_sel_logits = model(
                input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])
            batch_start_logits = batch_start_logits.detach().cpu().tolist()
            batch_end_logits = batch_end_logits.detach().cpu().tolist()
            batch_sel_logits = batch_sel_logits.detach().cpu().tolist()
            assert len(batch_start_logits)==len(batch_end_logits)==len(batch_sel_logits)
            for start_logit, end_logit, sel_logit in zip(batch_start_logits, batch_end_logits, batch_sel_logits):
                outputs.append((start_logit, end_logit, sel_logit))

    if save_predictions and dev_data.args.n_paragraphs is None:
        n_paragraphs = [dev_data.args.test_M]
    elif save_predictions:
        n_paragraphs = [int(n) for n in dev_data.args.n_paragraphs.split(",")]
    else:
        n_paragraphs = None
    predictions = dev_data.decode_span(outputs,
                                       n_paragraphs=n_paragraphs)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions, n_paragraphs=n_paragraphs))



fp = open("data/dpr_tokenized.json", "r")
tokenized_data = json.load(fp)
reader_dataset = AmbigQAReaderDataset(tokenized_data)
data_loader = ReaderDataLoader(reader_dataset)
fp.close()
reader = ReaderFactory().create_reader(
    reader_name=ReaderName.BERT.name,
    base="bert-base-uncased",
    checkpoint="C:/Users/deepa/Desktop/MSc CS/Research Assistantship/Project/AmbigQA/AmbigQA/codes/out/ambignq-span-selection/best-model.pt",
)
reader.eval()

inference_span_predictor(model=reader,dev_data=data_loader,save_predictions=True)

# with open("data/dpr_tokenized.json", "w") as f:
#     json.dump(tokenized_data, f)

