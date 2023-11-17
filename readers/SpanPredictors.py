import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertForQuestionAnswering, AlbertForQuestionAnswering

class BertSpanPredictor(BertForQuestionAnswering):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

    def forward(self,
                input_ids=None, attention_mask=None,
                token_type_ids=None, inputs_embeds=None,
                start_positions=None, end_positions=None, answer_mask=None,
                is_training=False):

        N, M, L = input_ids.size()
        output = self.bert(input_ids.view(N*M, L),
                           attention_mask=attention_mask.view(N*M, L),
                           token_type_ids=token_type_ids.view(N*M, L),
                           inputs_embeds=None if inputs_embeds is None else inputs_embeds.view(N*M, L, -1))[0]
        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        sel_logits = self.qa_classifier(output[:,0,:])
        return start_logits.view(N, M, L), end_logits.view(N, M, L), sel_logits.view(N, M)

class AlbertSpanPredictor(AlbertForQuestionAnswering):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

    def forward(self,
                input_ids=None, attention_mask=None,
                token_type_ids=None, inputs_embeds=None,
                start_positions=None, end_positions=None, answer_mask=None,
                is_training=False):

        N, M, L = input_ids.size()
        output = self.albert(input_ids.view(N*M, L),
                             attention_mask=attention_mask.view(N*M, L),
                             token_type_ids=token_type_ids.view(N*M, L),
                             inputs_embeds=None if inputs_embeds is None else inputs_embeds.view(N*M, L, -1))[0]
        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        sel_logits = self.qa_classifier(output[:,0,:])
        return start_logits.view(N, M, L), end_logits.view(N, M, L), sel_logits.view(N, M)

