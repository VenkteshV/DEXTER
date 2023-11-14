import os

import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch

from data.datastructures.arguments import TrainArgs


class Trainer:
    def __int__(self, model, optimizer: Optimizer, scheduler, loss_fn, logger):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.model = model

    def train(self, train_data: DataLoader, args: TrainArgs):
        self.model.train()
        global_step = 0
        train_losses = []
        best_accuracy = -1
        stop_training = False

        for _ in range(args.resume_global_step):
            self.optimizer.step()
            self.scheduler.step()

        self.logger.info("Start training!")
        for epoch in range(int(args.num_train_epochs)):
            for batch in train_data.dataloader:
                global_step += 1
                # batch = [b.to(torch.device("cuda")) for b in batch]
                self.model()
                # if args.is_seq2seq:
                #     loss = model(
                #         input_ids=batch[0],
                #         attention_mask=batch[1],
                #         decoder_input_ids=batch[2],
                #         decoder_attention_mask=batch[3],
                #         is_training=True,
                #     )
                # else:
                output = self.model(
                    input_ids=batch[0],
                    attention_mask=batch[1],
                    token_type_ids=batch[2],
                    start_positions=batch[3],
                    end_positions=batch[4],
                    answer_mask=batch[5],
                    is_training=True,
                )

                loss = loss(output, batch)
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if torch.isnan(loss).data:
                    self.logger.info("Stop training because loss=%s" % (loss.data))
                    stop_training = True
                    break
                train_losses.append(loss.detach().cpu())
                loss.backward()
