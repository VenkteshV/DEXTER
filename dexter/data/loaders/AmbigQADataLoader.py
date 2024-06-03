from dexter.config.constants import Split
from dexter.data.datastructures.answer import AmbigNQAnswer, Answer
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.evidence import Evidence
from dexter.data.datastructures.sample import AmbigNQSample
from dexter.data.loaders.BaseDataLoader import GenericDataLoader
import tqdm

class AmbigQADataLoader(GenericDataLoader):
    '''Data loader class to load Datset from raw AmbigQA dataset.
    AmbigQA dataset consists of a questions each of which has clarified sub questions which again each have their own answers. 
    Additionally each question has multiple annotations
    
    Arguments:
    dataset (str): string containing the dataset alias
    tokenzier (str) : name of the tokenizer model. Set tokenizer as None, if only samples to be loaded but not tokenized and stored. This can help save time if only the raw dataset is needed.
    config_path (str) : path to the configuration file containing various parameters
    split (Split) : Split of the dataset to be loaded
    batch_size (int) : batch size to process the dataset.
    corpus: corpus containing all needed passages.
    
    '''
    def __init__(
        self,
        dataset: str,
        tokenizer="bert-base-uncased",
        config_path="test_config.ini",
        split=Split.TRAIN,
        batch_size=None,
        corpus=None
    ):
        self.corpus = corpus
        self.titles = [self.corpus[idx].title() for idx,_ in enumerate(self.corpus)]
        print(self.titles[100],self.corpus[100].title(),len(self.titles))
        super().__init__(dataset, tokenizer, config_path, split, batch_size)

    def load_raw_dataset(self, split=Split.TRAIN):
        dataset = self.load_json(split)
        for index, data in enumerate(tqdm.tqdm(dataset)):
            _id = data["id"]
            question = Question(data["question"], str(index))
            sample_answers = []
            for annotation in data["annotations"]:
                if annotation["type"] == "singleAnswer":
                    answers = [
                        [Answer(answer, None) for answer in annotation["answer"]]
                    ]
                elif annotation["type"] == "multipleQAs":
                    answers = [
                        [Answer(answer, None) for answer in pair["answer"]]
                        for pair in annotation["qaPairs"]
                    ]
                else:
                    raise TypeError("Unknown annotation type: ", annotation["type"])
                sample_answers.append(answers)
            for article in data["viewed_doc_titles"]:
                    if article in self.titles:
                        idx_lookup = list(self.titles).index(article)
                        evidence = Evidence(text= self.corpus[idx_lookup].text(),
                                              title=article,
                                              idx=idx_lookup)

                    self.raw_data.append(
                                AmbigNQSample(_id, question, AmbigNQAnswer(sample_answers), evidences=evidence)
                            )

    def tokenize_answers(self, MAX_LENGTH=20):
        # tokenize answers and make list for each tokenized answer
        decoder_ids = []
        decoder_masks = []
        for sample in self.raw_data:
            sample_op_ids = []
            sample_op_attention_masks = []
            for answer in sample.answer.flatten():
                tokenized_ans = self.tokenizer.tokenize(
                    answer,
                    pad_to_max_length="bart" in self.tokenizer_name,
                    max_length=MAX_LENGTH,
                )
                sample_op_ids.append(tokenized_ans["input_ids"])
                sample_op_attention_masks.append(tokenized_ans["attention_mask"])
            decoder_ids.append(sample_op_ids)
            decoder_masks.append(sample_op_attention_masks)
        return decoder_ids, decoder_masks