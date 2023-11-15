"""DPR losses

Returns:
    _type_: _description_
"""
from torch import Tensor
import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from losses.BaseLoss import BaseLoss


class DprNllLoss(BaseLoss):
    def loss_fn(
        self,
        q_vectors: Tensor,
        ctx_vectors: Tensor,
        positive_idx_per_question: list,
        hard_negatice_idx_per_question: list = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.tensor(
                positive_idx_per_question).to(max_idxs.device)
        ).sum()
        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: Tensor, ctx_vectors: Tensor) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores
