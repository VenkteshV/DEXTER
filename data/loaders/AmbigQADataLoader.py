from constants import Split
from data.datastructures.answer import AmbigNQAnswer, Answer
from data.datastructures.question import Question
from data.datastructures.sample import AmbigNQSample
from data.loaders.BaseDataLoader import GenericDataLoader


class AmbigQADataLoader(GenericDataLoader):
    def __init__(
        self,
        dataset: str,
        tokenizer="bert-base-uncased",
        config_path="test_config.ini",
        split=Split.TRAIN,
        batch_size=None,
    ):
        super().__init__(dataset, tokenizer, config_path, split, batch_size)

    def load_raw_dataset(self, split=Split.TRAIN):
        dataset = self.load_json(split)
        for data in dataset:
            _id = data["id"]
            question = Question(data["question"], None)
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
            self.raw_data.append(
                AmbigNQSample(_id, question, AmbigNQAnswer(sample_answers))
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