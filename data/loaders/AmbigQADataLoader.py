from constants import Split
from data.datastructures.answer import AmbigNQAnswer, Answer
from data.datastructures.question import Question
from data.datastructures.evidence import Evidence
from data.datastructures.sample import AmbigNQSample
from data.loaders.BaseDataLoader import GenericDataLoader
import tqdm

class AmbigQADataLoader(GenericDataLoader):
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
                        evidence = Evidence(text= article,
                                              title=article,
                                              idx=(list(self.titles).index(article)))

                    self.raw_data.append(
                                AmbigNQSample(_id, question, AmbigNQAnswer(sample_answers), evidences=evidence)
                            )

    def tokenize_answers(self, MAX_LENGTH=20):
        # tokenize answers and make list for each tokenized answer
        samples = [sample.answer.flatten() for sample in self.raw_data]
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